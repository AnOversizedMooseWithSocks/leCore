"""holographic_vmplan.py -- FETCH/DECODE SEPARATED FROM EXECUTE for the holographic VM.

WHY THIS EXISTS
---------------
`HoloMachine.run()` decodes an instruction every time the program counter visits its address. Decoding
is the expensive half of the VM: one spectral read to pull address `i` out of the program vector, two
unbinds to peel OP and ARG off the instruction, and two nearest-atom cleanups against the codebooks.
Executing is the cheap half: a bind, a bundle, a roll.

But DECODE IS A PURE FUNCTION OF (program_vector, address). It never looks at the accumulator, the
register file or the stack. So re-deriving it on every visit is pure waste, and the waste compounds
exactly where programs get interesting:

  MEASURED (dim 1024, seed 7, `LOAD a; ITERATE step; HALT` with a 2-instruction body, max_loop=64):
      131 address decodes performed over 5 distinct addresses  ->  26.2x redundancy
      wall clock 40.4 ms for what is arithmetically 64 binds

That is the same observation that made real CPUs fast (an instruction cache in front of a decoder),
and it is lever 1 of the engine's own "walls are bad approaches" list -- bake once, sample O(1) --
pointed at the VM instead of at a texture.

WHAT THIS MODULE ADDS
---------------------
`DecodePlan` decodes a WHOLE BLOCK OF ADDRESSES IN ONE BATCHED SPECTRAL SWEEP and memoises the result
per program, so:
  * the L address reads become 1 forward transform + 1 batched inverse transform instead of 2L
    transforms, because the program's spectrum is multiplied against a resident (L, D/2+1) stack of
    pre-transformed address keys;
  * the 2L role unbinds become 2 batched transforms instead of 6L;
  * the 2L codebook cleanups become a couple of (L, D) @ (D, K) matmuls instead of 2L Python loops;
  * every LATER visit to an address -- the loop body, the called function, the re-run of the same
    program -- costs a dict lookup.

CORRECTNESS: BIT-IDENTICAL BY CONSTRUCTION, NOT BY SAMPLING
-----------------------------------------------------------
Two separate claims, each earned rather than asserted:

  1. The SPECTRAL half is bit-identical. A batched `np.fft.rfft(X, axis=-1)` returns exactly the same
     bytes as stacking per-row transforms (verified at D = 256 / 1024 / 4096, `array_equal` True), and
     the address keys are cached as `rfft(involution(pos(i)))` -- the identical array `_read_addr`
     builds -- NOT as `conj(rfft(pos(i)))`. The conjugate identity is mathematically true but only
     numerically true to ~5e-16, and a decode path that has to argue about epsilons is a decode path
     that will eventually flip a tie on somebody else's BLAS. So we do not take the shortcut.

  2. The CLEANUP half is symbol-identical. Batched `(L, D) @ (D, K)` scoring regroups the float
     summation relative to `_nearest_loop`'s per-atom cosine, and the 1-ULP DIRECTION of that
     regrouping differs across BLAS builds -- this exact knife-edge already bit CI once, on
     `_nearest_fast` (an exact LOAD/BIND tie flipped). The fix there was structural and it is reused
     verbatim here: wherever the top scores of a row sit inside a 1e-9 band, JUST those candidates go
     back to `_nearest_loop`'s exact arithmetic in codebook order. Agreement then holds by
     construction on any BLAS, not where we happened to sample.

KEPT NEGATIVES (loud, so nobody re-derives them)
------------------------------------------------
  * THE FUSED DECODE IS DELIBERATELY NOT SHIPPED. You can skip the intermediate irfft/rfft round trip
    entirely (`irfft(P * conj_pos * conj_OP)`), which drops the transform count from 3 batched to 2
    and measured a further ~1.15x. It agrees on every symbol we tested and differs from the reference
    by 5e-17 in the raw vector -- and that is precisely the "hammered N cases, zero disagreements"
    argument the QEM lesson says is a SAMPLE, not a proof. Filed as possible-but-not-taken, not as
    impossible; reopen it only with a structural argument, not more samples.
  * CACHING BY `id()` IS UNSAFE HERE and was tried first. CALL and ITERATE rebuild the callee body
    with `unbind(library, fn_atom)` on every iteration, so the body is a FRESH array with a fresh id
    each time -- an identity key misses 100% of the time on exactly the workload the cache exists for.
    Content hashing is what makes those hits land, and its cost (one sha256 of the program vector per
    `decode` call, ~21 us at D=1024) is amortised over a whole block of addresses rather than paid per
    address. That ratio is the entire design.
  * THE SPECTRUM CACHE NEXT DOOR DOES NOT HELP HERE, and measuring it is how this module found its own
    key discipline: `holographic_residency.SpectrumCache` hashes its operand on EVERY lookup, and
    sha256 of a D-float atom costs more than the rfft it is avoiding (D=1024: 21.5 us hash vs 13.0 us
    transform; D=4096: 85.5 us vs 36.2 us). Its own docstring claims ~1.4x; re-measured on a 100%-warm
    cache it is 0.40x-0.80x, i.e. a slowdown. Hence: this module hashes ONCE PER BLOCK and keys its
    resident address spectra by INTEGER ADDRESS, which is free.

Pure NumPy + stdlib + hashlib. Deterministic. Opt-in: `HoloMachine(decode_plan=True)`.
"""

import hashlib
from collections import OrderedDict

import numpy as np

from holographic.sampling_and_signal.holographic_fft import rfft as _rfft, irfft as _irfft
from holographic.agents_and_reasoning.holographic_ai import involution

# How many addresses a single batched sweep decodes. Programs are small (the HRR capacity cliff caps a
# single program vector at ~20-32 instructions at dim 1024), so one block of 32 usually covers the whole
# thing and the block cost is paid exactly once. Overshooting past HALT is harmless -- those rows decode
# to noise, we stop consuming at the first HALT -- and it is cheaper than a second sweep.
DEFAULT_BLOCK = 32

# The near-tie band, in cosine units. Identical to _nearest_fast's, and for the identical reason: it is
# wide enough to catch any float-regrouping disagreement (which live at ~1e-16) and far narrower than
# any real margin between two distinct random atoms (which live at ~1/sqrt(D), i.e. ~0.03 at D=1024).
TIE_BAND = 1e-9


def program_key(program_vec):
    """Content hash of a program vector -- the cache key. hashlib, never Python's hash(), so the key is
    identical across processes and across PYTHONHASHSEED settings.

    Content addressing (not identity) is load-bearing: CALL and ITERATE rebuild the callee's body vector
    from the library on every single iteration, so two byte-identical bodies must land on the same key or
    the cache never hits on the workload it exists for."""
    a = np.ascontiguousarray(np.asarray(program_vec, float))
    return hashlib.sha256(a.tobytes()).hexdigest()


class DecodePlan:
    """A decoded-instruction cache in front of a HoloMachine: decode a block of addresses once, in one
    batched spectral sweep, then answer every later visit from memory.

    Attach one to a machine (`HoloMachine(decode_plan=True)` does it for you) and call `at(prog, pc)`
    wherever the interpreter would have called `decode_instruction(prog, pc)`. The answers are identical;
    the second and every subsequent visit is a dict lookup.

    Cheap to hold: the resident address spectra are shared across every program of the same dimension,
    and the per-program entry is a small dict of (opcode, operand) name pairs -- text, not vectors."""

    def __init__(self, machine, block=DEFAULT_BLOCK, max_programs=64):
        self.m = machine
        self.dim = int(machine.dim)
        self.block = int(block)
        self.max_programs = int(max_programs)
        # RESIDENT ADDRESS KEYS: rfft(involution(pos(i))), keyed by the INTEGER address. Free to key
        # (an int), pure to derive (seed + dim + i), and shared by every program this machine runs --
        # so the transform cost of the address alphabet is paid once per machine, not once per decode.
        self._addr_spec = {}
        self._role_spec = {}                 # role name -> rfft(involution(role)); OP and ARG, same deal
        self._books = {}                     # codebook id -> (names, row-normalised matrix)
        self._cache = OrderedDict()          # program hash -> {address: (op, arg)}, LRU
        self._key_memo = OrderedDict()       # id(array) -> (array, hash); see _key -- holds a strong ref
        self.hits = 0
        self.misses = 0
        self.sweeps = 0                      # how many batched decodes actually ran (the thing we minimise)

    def _key(self, program_vec):
        """The cache key for a program vector: an identity memo in front of the content hash.

        WHY BOTH. The content hash is what makes CALL/ITERATE hit at all (each iteration rebuilds the
        callee body as a fresh array, so an identity-only key misses every time). But hashing 8 KB on
        EVERY lookup then dominates a tight loop -- measured 21 us at D=1024, against a planned decode
        of a few us. So: hash once per distinct array OBJECT, remember the answer under its id, and
        note that we keep a STRONG REFERENCE to the array in the memo -- that is what makes id() a
        legitimate key here, because CPython cannot recycle the id of an object we are still holding.
        Two byte-identical arrays with different ids both hash, both land on the same content key, and
        both hit the same decoded entry: correctness comes from the hash, speed from the memo.

        THE ONE ASSUMPTION, stated loudly: a program vector is never mutated in place after assembly.
        Every construction path builds a fresh array (assemble, unbind out of the library), so this
        holds -- but an in-place edit of a cached program would serve a stale decode. Call clear()."""
        k = id(program_vec)
        hit = self._key_memo.get(k)
        if hit is not None and hit[0] is program_vec:
            return hit[1]
        key = program_key(program_vec)
        self._key_memo[k] = (program_vec, key)
        if len(self._key_memo) > 4 * self.max_programs:
            self._key_memo.popitem(last=False)
        return key

    # -- resident spectra -------------------------------------------------------------------------
    def _addr_stack(self, lo, hi):
        """The (hi-lo, D/2+1) stack of pre-transformed address keys for addresses lo..hi-1.

        Each row is exactly the array `_read_addr` multiplies against, so the batched product is the
        batched form of the same arithmetic -- not an algebraically-equal rewrite of it."""
        for i in range(lo, hi):
            if i not in self._addr_spec:
                self._addr_spec[i] = _rfft(involution(self.m.pos(i)))
        return np.stack([self._addr_spec[i] for i in range(lo, hi)])

    def _role(self, name, vec):
        """rfft(involution(role)) for a role atom, cached by role NAME. Two roles exist (OP, ARG) and
        they are built once in the machine's __init__, so this is a two-entry table for the life of
        the machine."""
        s = self._role_spec.get(name)
        if s is None:
            s = _rfft(involution(vec))
            self._role_spec[name] = s
        return s

    def _book(self, table):
        """(names, row-normalised matrix) for a codebook, cached by the table object's id.

        Identity is a safe key here for the reason it is NOT safe for programs: the machine's codebooks
        are built once in __init__ and never mutated or rebuilt, so their ids cannot be recycled while
        the machine is alive. The length is re-checked so a grown table (define() adds function atoms)
        rebuilds instead of silently serving a stale matrix."""
        key = id(table)
        entry = self._books.get(key)
        if entry is None or entry[0] != len(table):
            names = list(table.keys())
            M = np.stack([table[n] for n in names])
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-300)
            entry = (len(table), names, Mn)
            self._books[key] = entry
        return entry[1], entry[2]

    # -- the batched classifier -------------------------------------------------------------------
    def _classify(self, rows, table):
        """Nearest atom in `table` for each row of `rows`, as a list of names.

        One (L, D) @ (D, K) matmul replaces L Python loops over K cosine calls. Rows whose top scores
        sit inside TIE_BAND are handed back to the machine's own exact `_nearest_loop` over just those
        candidates, in codebook order -- so a regrouped-float tie can never flip a symbol, on any BLAS.
        This is the same structural guarantee _nearest_fast earned after CI sampled the knife edge."""
        names, Mn = self._book(table)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        # A zero row has all-zero cosines; _nearest_loop's strict > keeps the FIRST entry, so match that.
        safe = np.where(norms > 0, norms, 1.0)
        scores = (rows / safe) @ Mn.T
        top = np.argmax(scores, axis=1)
        out = []
        for r in range(scores.shape[0]):
            if norms[r, 0] == 0.0:
                out.append(names[0])
                continue
            band = np.flatnonzero(scores[r] >= scores[r, top[r]] - TIE_BAND)
            if len(band) > 1:
                sub = {names[i]: table[names[i]] for i in sorted(band)}
                out.append(self.m._nearest_loop(sub, rows[r]))
            else:
                out.append(names[int(top[r])])
        return out

    def _operand_table(self, op):
        """The codebook an opcode's operand must be cleaned against. One place, so the plan and the
        machine's own `decode_instruction` can never drift on the opcode->codebook mapping."""
        m = self.m
        if op in ("CALL", "ITERATE") and m.fn_atoms:
            return m.fn_atoms
        if op == "APPLY":
            return m.fac_atoms
        if op == "REPEAT":
            return m.cnt_atoms
        if op in ("STORE", "RECALL"):
            return m.reg_atoms
        return m.data_atoms

    # -- the sweep --------------------------------------------------------------------------------
    def sweep(self, program_vec, lo, hi):
        """Decode addresses lo..hi-1 of `program_vec` in one batched pass. Returns {address: (op, arg)}.

        Three batched transforms total, regardless of how many addresses are in the block:
          1  rfft(program_vec)                       -- the program's spectrum, once
          2  irfft(P * ADDR, axis=-1)                -- every address read, at once
          3  irfft(rfft(raw) * ROLE, axis=-1) x2     -- both role peels, at once
        against 8 scalar transforms PER ADDRESS in the unplanned path."""
        self.sweeps += 1
        D = self.dim
        P = _rfft(np.asarray(program_vec, float))
        raw = _irfft(P[None, :] * self._addr_stack(lo, hi), n=D, axis=-1)      # (L, D) -- same bytes as _read_addr
        R = _rfft(raw, axis=-1)
        op_rows = _irfft(R * self._role("OP", self.m.OP), n=D, axis=-1)
        arg_rows = _irfft(R * self._role("ARG", self.m.ARG), n=D, axis=-1)
        ops = self._classify(op_rows, self.m.op_atoms)
        # OPERANDS GROUPED BY CODEBOOK: which codebook an operand belongs to depends on its own opcode,
        # so we bucket the rows by table and run one matmul per distinct table -- usually one or two,
        # never one per row. Same answers as decoding each address alone, a fraction of the calls.
        buckets = {}
        for k, op in enumerate(ops):
            buckets.setdefault(id(self._operand_table(op)), (self._operand_table(op), []))[1].append(k)
        args = [None] * len(ops)
        for table, idxs in buckets.values():
            for k, nm in zip(idxs, self._classify(arg_rows[idxs], table)):
                args[k] = nm
        return {lo + k: (ops[k], args[k]) for k in range(len(ops))}

    # -- the cached front door --------------------------------------------------------------------
    def decode(self, program_vec, upto=None):
        """Decoded instructions for `program_vec` as a dict {address: (op, arg)}, sweeping only if the
        requested address is not already known. `upto` forces coverage of at least that address."""
        key = self._key(program_vec)
        entry = self._cache.get(key)
        if entry is None:
            entry = {}
            self._cache[key] = entry
            if len(self._cache) > self.max_programs:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        want = self.block - 1 if upto is None else int(upto)
        if want not in entry:
            lo = 0 if not entry else max(entry) + 1
            hi = max(want + 1, lo + self.block)
            entry.update(self.sweep(program_vec, lo, hi))
        return entry

    def at(self, program_vec, pc):
        """(opcode, operand) at address `pc` -- the drop-in replacement for `decode_instruction`.

        Identical answers; the second and every later visit to an address costs a dict lookup instead of
        eight transforms and two codebook loops."""
        pc = int(pc)
        key = self._key(program_vec)
        entry = self._cache.get(key)
        if entry is not None and pc in entry:
            self._cache.move_to_end(key)
            self.hits += 1
            return entry[pc]
        self.misses += 1
        return self.decode(program_vec, upto=pc)[pc]

    def stats(self):
        """Cache telemetry: hits, misses, batched sweeps actually performed, programs resident.

        `sweeps` is the number to watch -- it is the count of times real spectral work happened, and on a
        loop-heavy program it should stay in the single digits no matter how many iterations run."""
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "sweeps": self.sweeps,
                "programs": len(self._cache),
                "hit_rate": (self.hits / total) if total else 0.0}

    def clear(self):
        """Drop the decoded-program cache (the resident address/role spectra and codebooks are kept --
        they are pure functions of the machine's format and can never go stale)."""
        self._cache.clear()
        self._key_memo.clear()
        self.hits = self.misses = self.sweeps = 0
        return self


def _selftest():
    """The two contracts, asserted numerically and loudly:

      A. The plan's (op, arg) is IDENTICAL to the machine's own scalar decode, over every address of a
         corpus of programs at three dimensions and two seeds -- including the noise addresses past HALT,
         which is where a decode disagreement would show up first.
      B. The plan does the work ONCE: re-decoding the same program leaves `sweeps` unchanged, and the
         batched read is bit-identical (array_equal, not allclose) to the scalar `_read_addr`."""
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine

    programs = [
        [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("PERMUTE", None), ("HALT", None)],
        [("LOAD", "a"), ("STORE", "R3"), ("LOAD", "d"), ("BIND", "e"), ("RECALL", "R3"), ("HALT", None)],
        [("LOAD", "b"), ("IFMATCH", "b"), ("BIND", "c"), ("PUSH", None), ("POP", None), ("HALT", None)],
        [("LOAD", "a"), ("REPEAT", 3), ("BIND", "f"), ("HALT", None)],
    ]
    checked = 0
    for dim in (256, 1024, 2048):
        for seed in (7, 11):
            m = HoloMachine(dim=dim, seed=seed)
            m.define("body", [("BIND", "b"), ("HALT", None)])
            plan = DecodePlan(m)
            for prog in programs + [[("LOAD", "a"), ("ITERATE", "body"), ("HALT", None)]]:
                pv = m.assemble(prog)
                # (A) every address, INCLUDING past the end where the read is pure crosstalk noise
                for i in range(len(prog) + 4):
                    ref = m.decode_instruction(pv, i)
                    got = plan.at(pv, i)
                    assert ref == got, f"decode disagreement dim={dim} seed={seed} addr={i}: {ref} vs {got}"
                    checked += 1
                # (B) bit-identity of the batched spectral read against the scalar one
                spec = _rfft(pv)
                batched = _irfft(spec[None, :] * plan._addr_stack(0, 4), n=dim, axis=-1)
                for i in range(4):
                    scalar = m._read_addr(spec, i, dim)
                    assert np.array_equal(batched[i], scalar), \
                        f"batched read is not bit-identical at dim={dim} addr={i}"
    assert checked >= 250, f"corpus too small to be a regression trap ({checked} comparisons)"

    # (B continued) the whole point: work happens once, not once per visit.
    m = HoloMachine(dim=1024, seed=7)
    plan = DecodePlan(m)
    pv = m.assemble([("LOAD", "a"), ("BIND", "b"), ("HALT", None)])
    plan.at(pv, 0)
    first = plan.sweeps
    assert first == 1, f"expected exactly one sweep to cover the block, got {first}"
    for _ in range(200):
        for pc in (0, 1, 2):
            plan.at(pv, pc)
    assert plan.sweeps == first, f"re-visits caused {plan.sweeps - first} extra sweep(s) -- the cache is not holding"
    st = plan.stats()
    assert st["hit_rate"] > 0.99, f"hit rate collapsed to {st['hit_rate']:.3f}"

    # A DIFFERENT program must not be served from another program's entry (content addressing, not luck).
    pv2 = m.assemble([("LOAD", "d"), ("BUNDLE", "e"), ("HALT", None)])
    assert plan.at(pv2, 0) == ("LOAD", "d"), "content key collided across distinct programs"
    assert plan.sweeps == first + 1, "a new program should have cost exactly one more sweep"

    print(f"holographic_vmplan selftest OK -- {checked} decode comparisons identical, "
          f"batched read bit-identical, {plan.stats()}")


if __name__ == "__main__":
    _selftest()
