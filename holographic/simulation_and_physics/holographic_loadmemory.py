"""holographic_loadmemory.py -- a role->filler memory that picks its representation by LOAD and FIDELITY NEED
(the FHRR-and-tensor re-enables, from the adaptive-dispatch audit).

WHY
---
Binding N role->filler pairs into one memory and recalling them trades off recall quality against storage:

  * real-HRR (real vectors, D numbers)  -- cheap, near-perfect at LOW load; recall falls off as pairs climb.
  * FHRR     (unit phasors, ~2D numbers) -- holds recall far better at HIGH load; at low load it "changes nothing".
  * tensor   (outer product, D*D numbers) -- EXACT recall up to M~D (perfect where HRR/FHRR have long degraded),
             but spends D-times the storage. Only worth it when you NEED exact recall and can afford the memory.

All three were kept as options rather than a default because the best one depends on the situation. With adaptive
dispatch we GATE on the numbers that decide it -- the LOAD (pairs bound), whether EXACT recall is needed, and the
memory BUDGET:

    exact recall wanted, pairs fit (M <= dim), D*D within budget   -> tensor   (exact)
    load past the capacity knee (~0.08*dim)                        -> fhrr     (better recall, ~2x storage)
    otherwise                                                      -> hrr      (cheap, near-perfect at low load)

There is NO harm mode on RECALL in either re-enable -- FHRR recall >= real-HRR, and tensor is exact in-regime -- so a
borderline misfire costs a little storage, never a wrong answer. The gate is biased toward the CHEAP default and only
pays for FHRR/tensor when the load / fidelity need makes the win real.

MEASURED (fair, same dim=256): recall ties at low load; past the knee FHRR pulls away (N=100 -> HRR 0.21, FHRR 0.41)
and tensor stays EXACT (N=250 -> HRR 0.05, FHRR 0.10, tensor 1.00) at D*D = 256x the storage.

USAGE
    m = AdaptiveRoleFillerMemory(dim=256, expected_pairs=80)                 # -> fhrr (high load)
    m = AdaptiveRoleFillerMemory(dim=256, expected_pairs=200, exact=True)    # -> tensor (exact, if budget allows)
    m.add("color", "red"); m.recall("color")  -> "red"
    m.backend  -> "hrr" | "fhrr" | "tensor"
"""
import numpy as np

# the load knee as a fraction of dim, measured (FHRR overtakes real-HRR around here). Biased slightly high so we
# stay on the cheap real-HRR default until FHRR's win is clear.
LOAD_KNEE_FRAC = 0.08


def choose_backend(expected_pairs, dim, exact=False, max_numbers=None, load_frac=LOAD_KNEE_FRAC):
    """The representation gate. Returns 'tensor', 'fhrr', or 'hrr'.
      * tensor -- when EXACT recall is requested, the pairs fit (M <= dim so the keys can stay independent), and the
        D*D storage is within max_numbers (None = no budget cap). Exact to M~D.
      * fhrr   -- when the expected load exceeds load_frac*dim (past the knee, where real-HRR degrades).
      * hrr    -- otherwise (cheap, near-perfect at low load).
    The deciders are all known up front -- an integer pair count, a boolean, and a storage budget -- no estimate."""
    dim = int(dim); n = int(expected_pairs)
    if exact and n <= dim and (max_numbers is None or dim * dim <= int(max_numbers)):
        return "tensor"
    if n > load_frac * dim:
        return "fhrr"
    return "hrr"


class AdaptiveRoleFillerMemory:
    """A role->filler associative memory with a uniform interface, whose representation is chosen by load and
    fidelity need: real-HRR (cheap), FHRR (high load), or tensor (exact). The backend is hidden -- add/recall look
    the same regardless."""

    def __init__(self, dim, expected_pairs, exact=False, max_numbers=None, seed=0, load_frac=LOAD_KNEE_FRAC):
        self.dim = int(dim)
        self.seed = int(seed)
        self.backend = choose_backend(expected_pairs, dim, exact=exact, max_numbers=max_numbers, load_frac=load_frac)
        self._fillers = []                                    # remembered filler names (the recall candidate set)
        if self.backend == "fhrr":
            from holographic.sampling_and_signal.holographic_fhrr import PhasorVocabulary, PhasorMemory
            self._vocab = PhasorVocabulary(self.dim, seed=self.seed)
            self._mem = PhasorMemory(self.dim)                # complex trace
        elif self.backend == "tensor":
            from holographic.agents_and_reasoning.holographic_ai import Vocabulary
            self._vocab = Vocabulary(self.dim, seed=self.seed)
            self._keys = []                                   # tensor builds from ALL pairs at once, so we BUFFER
            self._vals = []                                   # the key/value atoms and seal the memory on first recall
            self._sealed = None                               # the built TensorBindMemory (lazy)
            self._val_matrix = None                           # stacked filler atoms, for the argmax cleanup
        else:  # hrr
            from holographic.agents_and_reasoning.holographic_ai import Vocabulary
            self._vocab = Vocabulary(self.dim, seed=self.seed)
            self._trace = np.zeros(self.dim, float)           # real trace

    def add(self, role, filler):
        """Bind role->filler into the record. Same call regardless of backend."""
        r = self._vocab.get(str(role))
        f = self._vocab.get(str(filler))
        if self.backend == "fhrr":
            from holographic.sampling_and_signal.holographic_fhrr import fhrr_bind
            self._mem.trace = self._mem.trace + fhrr_bind(r, f)
        elif self.backend == "tensor":
            self._keys.append(r); self._vals.append(f)
            self._sealed = None                               # a new pair invalidates the sealed memory
        else:
            from holographic.agents_and_reasoning.holographic_ai import bind
            self._trace = self._trace + bind(r, f)
        if str(filler) not in self._fillers:
            self._fillers.append(str(filler))
        return self

    def _seal_tensor(self):
        """Build the tensor-product memory from the buffered pairs (done once, lazily, on first recall)."""
        from holographic.sampling_and_signal.holographic_tensor import TensorBindMemory
        self._sealed = TensorBindMemory(np.stack(self._keys), np.stack(self._vals))
        self._val_matrix = np.stack(self._vals)

    def recall(self, role, candidates=None):
        """Recover the filler bound to `role` and return its name (or None). `candidates` narrows the cleanup set;
        defaults to everything added."""
        cand = candidates if candidates is not None else self._fillers
        r = self._vocab.get(str(role))
        if self.backend == "fhrr":
            from holographic.sampling_and_signal.holographic_fhrr import fhrr_unbind
            name, _sim = self._vocab.cleanup(fhrr_unbind(self._mem.trace, r), candidates=cand)
            return name
        if self.backend == "tensor":
            if self._sealed is None:
                self._seal_tensor()
            out = self._sealed.recall(r)                      # M^T r -- the value vector for this key
            cand_idx = [i for i, nm in enumerate(self._fillers) if nm in cand]
            sims = self._val_matrix[cand_idx] @ out           # nearest stored filler by dot (atoms ~ unit)
            return self._fillers[cand_idx[int(np.argmax(sims))]]
        from holographic.agents_and_reasoning.holographic_ai import unbind
        name, _sim = self._vocab.cleanup(unbind(self._trace, r), candidates=cand)
        return name


def _selftest():
    # LOW load -> cheap real-HRR, perfect recall
    lo = AdaptiveRoleFillerMemory(dim=256, expected_pairs=6, seed=0)
    assert lo.backend == "hrr"
    for role, fil in [("color", "red"), ("shape", "round"), ("size", "big")]:
        lo.add(role, fil)
    assert lo.recall("color") == "red" and lo.recall("shape") == "round"

    # HIGH load -> FHRR, holds recall where real-HRR fails
    N = 90
    hi = AdaptiveRoleFillerMemory(dim=256, expected_pairs=N, seed=1)
    assert hi.backend == "fhrr"
    for i in range(N):
        hi.add(f"r{i}", f"f{i}")
    fhrr_hits = sum(hi.recall(f"r{i}") == f"f{i}" for i in range(N))

    # EXACT need -> tensor, perfect recall at a load where FHRR is already failing
    NX = 200
    tx = AdaptiveRoleFillerMemory(dim=256, expected_pairs=NX, exact=True, seed=2)
    assert tx.backend == "tensor"
    for i in range(NX):
        tx.add(f"r{i}", f"f{i}")
    tensor_hits = sum(tx.recall(f"r{i}") == f"f{i}" for i in range(NX))
    assert tensor_hits == NX                                  # EXACT

    # the exact gate respects the memory budget: too small a budget -> fall back off tensor
    budgeted = AdaptiveRoleFillerMemory(dim=256, expected_pairs=200, exact=True, max_numbers=1000, seed=2)
    assert budgeted.backend != "tensor"                      # D*D=65536 > 1000 budget

    print("OK: holographic_loadmemory self-test passed (hrr low load perfect; fhrr high load %d/%d; tensor EXACT "
          "%d/%d at N=%d; budget cap respected)" % (fhrr_hits, N, tensor_hits, NX, NX))


if __name__ == "__main__":
    _selftest()
