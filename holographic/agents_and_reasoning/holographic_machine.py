"""Holographic stored-program machine -- the 'operating system' rung of the substrate.

WHY THIS EXISTS
---------------
A hard drive has structure (the platter), data laid down in that structure (magnetization), and --
when that data is read back and executed -- a whole new layer of structure (an OS, a VM, an OS inside
the VM). holostuff already had the first rungs: the D-dim vector is the platter, derived_atom(seed,...)
is the low-level format, and role-filler binding + nested composition is the file system. The rung this
module adds is the one that makes the tower possible: a PROGRAM encoded as a hypervector, executed by
the engine's own bind/bundle/cleanup operations. Instructions and data live in the same vector space
(von Neumann, holographically), so the thing that stores structure can also store the recipe for MORE
structure -- and run it.

"FORMATTING THE DRIVE" is just fixing a seed: that deterministically lays down the alphabet -- two
roles (OP, ARG), the opcode atoms, the data atoms, an address function POS(i), and a SLOT role for
nesting. Same seed => bit-identical format on any machine.

THE INSTRUCTION SET (deliberately tiny and readable)
    LOAD x   : ACC = x                  (put a value in the accumulator)
    BIND x   : ACC = bind(ACC, x)       (associate -- the 'multiply' of the algebra)
    BUNDLE x : ACC = bundle([ACC, x])   (superpose -- the 'add')
    PERMUTE  : ACC = permute(ACC, 1)    (shift -- encodes order/position)
    HALT     : stop
A program is a list of (opcode, operand). assemble() encodes it as ONE vector:
    instruction_i = bundle(bind(OP, opcode_i), bind(ARG, operand_i))
    program       = bundle_i( bind(POS(i), instruction_i) )
run() reads each address by unbinding POS(i), CLEANS the opcode and operand against their codebooks
(wide-margin classification -- robust to the bundling crosstalk), and dispatches. Because operands are
cleaned to exact atoms before use, the accumulator is built from clean atoms and is EXACT even though
the program-reading itself is noisy.

MEASURED (honest picture, kept negatives included)
  * Correctness: 'LOAD a; BIND b; BUNDLE c' yields ACC == bundle(bind(a,b),c) at cosine 1.0000.
  * DRIVE SIZE (capacity cliff): instruction-decode holds ~100% up to a program length that scales
    with dimension -- ~32 instructions reliable at dim 1024, ~128 at dim 4096 -- then bundling
    crosstalk overwhelms cleanup and accuracy falls. The cliff is real: capacity is finite. (KEPT
    NEGATIVE -- this is the honest HRR capacity wall, not hidden.)
  * INCEPTION DEPTH (nesting a program as a file inside a disk inside a disk...): effectively
    UNBOUNDED when each level is clean (depth 8+ runs fine -- a pure unitary bind chain barely
    degrades), but bounded to ~3-4 levels when each disk also holds OTHER files. The law: you can
    nest as deep as you like if each level is uncluttered; a busy disk corrupts a buried program
    after a few levels. Both numbers scale with dimension.

Pure NumPy, deterministic, no new dependencies.
"""

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, cosine, permute, derived_atom, bind_batch, involution

OPCODES = ["LOAD", "BIND", "BUNDLE", "PERMUTE", "CALL", "APPLY", "IFMATCH", "ITERATE", "REPEAT", "HALT",
           "STORE", "RECALL",                                  # STORE r / RECALL r: a small register file (ISA-4)
           "PUSH", "POP"]                                      # PUSH / POP: the permute-stack for nesting (ISA-5)
DEFAULT_DATA = list("abcdef")
COUNT_MAX = 8                      # REPEAT's count operand ranges over cnt:1 .. cnt:COUNT_MAX
REGISTERS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]   # a handful of named slots beyond ACC (ISA-4)
# Faculty names an APPLY instruction can name. APPLY <faculty> means ACC := faculty(ACC) -- a unary
# map run by a host (the mind) that supplies the handlers; the bare VM has none, so APPLY is a no-op
# here. This is the extension point that lets a procedure invoke the engine's faculties as steps.
DEFAULT_FACULTIES = ["cleanup", "denoise", "matmul"]


# ---- the permute-stack (ISA-5): a LIFO stack in the vector substrate -----------------------------------------
# push is permute+bundle (shift the existing items one level deeper, drop the new item on top); pop is
# cleanup+inverse-permute (the top is the un-permuted item, so clean it out, peel it off, un-shift the rest).
# It is a genuine stack, but a HOLOGRAPHIC one: every level rides the same bundle, so depth is bounded by
# crosstalk -- deep stacks blur exactly like the B8 iterated-decode cliff (safe depth ~4-8 at dim 1024). Use it
# for shallow nesting of cleanup-able items; for arbitrary intermediates at any depth, use the (exact) registers.
def stack_push(stack, x):
    """Push x onto the permute-stack. The existing stack is permuted one step (pushed deeper); x lands on top."""
    return x if stack is None else bundle([x, permute(stack, 1)])


def stack_pop(stack, codebook):
    """Pop the top of the permute-stack, cleaning it against `codebook`. Returns (top_vec, remaining_stack).
    The top is the only un-permuted term, so cleanup recovers it; subtracting it and inverse-permuting restores
    the stack one level shallower. Exact for codebook items until the depth cliff."""
    top = max(codebook, key=lambda c: cosine(stack, c))      # the un-permuted item wins -- that is the top
    rest = permute(stack - top, -1)                          # peel the top off and un-shift the remaining stack
    return top, rest


class HoloMachine:
    """A formatted holographic drive that can store and execute stored programs."""

    def __init__(self, dim=4096, seed=7, data=None, faculties=None, fast_cleanup=False):
        self.dim = dim
        self.seed = seed
        # OPT-IN SIMD cleanup (see _nearest_fast): False keeps the original Python-loop cleanup, the recorded
        # decision. True routes every decode through one cached-codebook matmul -- measured 9x per decode.
        self.fast_cleanup = bool(fast_cleanup)
        self._cleanup_mats = {}                                # id(table) -> (len, names, normalised matrix)
        self._atom_cache = {}                                  # (name, unitary) -> derived atom; pure, bit-identical
        self.data_names = list(data) if data is not None else list(DEFAULT_DATA)
        self.faculty_names = list(faculties) if faculties is not None else list(DEFAULT_FACULTIES)
        # "format the drive": the whole alphabet is derived deterministically from the seed.
        self.OP = self._atom("role:OP", unitary=True)     # roles are unitary -> unbind is exact
        self.ARG = self._atom("role:ARG", unitary=True)
        self.SLOT = self._atom("role:SLOT", unitary=True)  # the role a nested 'file' lives under
        self.ACC = self._atom("role:ACC", unitary=True)    # the accumulator's role in a STATE snapshot (continuation)
        self.STK = self._atom("role:STK", unitary=True)    # the permute-stack's role in a state snapshot
        self.op_atoms = {o: self._atom(f"op:{o}") for o in OPCODES}
        self.data_atoms = {d: self._atom(f"dat:{d}") for d in self.data_names}
        self.fac_atoms = {f: self._atom(f"fac:{f}") for f in self.faculty_names}  # APPLY's operand codebook
        self.cnt_atoms = {n: self._atom(f"cnt:{n}") for n in range(1, COUNT_MAX + 1)}  # REPEAT's count codebook
        self.reg_atoms = {r: self._atom(f"reg:{r}") for r in REGISTERS}  # STORE/RECALL's register-name codebook
        # the holographic function LIBRARY: named sub-programs, all held in one vector, callable by name
        self.functions = {}       # name -> assembled program vector
        self.fn_atoms = {}        # name -> unitary name atom (the 'address' of the function)
        self.library = None       # bundle over names of bind(name_atom, program) -- the whole library, one vector

    def _atom(self, name, unitary=False):
        # MEMOIZED: derived_atom is a pure function of (seed, name, dim, unitary) -- same bytes every call -- and
        # the profile showed the VM re-deriving pos:i / role atoms on EVERY decode (940 FFT calls per 90
        # instructions, the actual hot path; the cleanup loop was only 12% of it). Caching a pure derivation is
        # bit-identical by construction: the cache returns the SAME array the first call produced. Callers never
        # mutate atoms (they bind/bundle into fresh arrays), so sharing the object is safe.
        key = (name, unitary)
        v = self._atom_cache.get(key)
        if v is None:
            v = derived_atom(self.seed, name, self.dim, unitary=unitary)
            self._atom_cache[key] = v
        return v

    def pos(self, i):
        """Address of the i-th instruction -- a deterministic unitary 'cylinder' atom."""
        return self._atom(f"pos:{i}", unitary=True)

    # ---- assembling a program into a single hypervector --------------------------------------
    def _instr(self, op, arg):
        if op in ("CALL", "ITERATE"):
            arg_vec = self.fn_atoms[arg]                            # operand is a function NAME (callee / loop body)
        elif op == "APPLY":
            arg_vec = self.fac_atoms[arg]                          # APPLY's operand is a faculty NAME
        elif op == "REPEAT":
            arg_vec = self.cnt_atoms[arg]                          # REPEAT's operand is a small-integer count
        elif op in ("STORE", "RECALL"):
            arg_vec = self.reg_atoms[arg]                          # STORE/RECALL's operand is a register name
        else:
            arg_vec = self.data_atoms.get(arg, self.op_atoms["HALT"])   # IFMATCH/value operand is a data atom
        return bundle([bind(self.OP, self.op_atoms[op]), bind(self.ARG, arg_vec)])

    def define(self, name, program):
        """Embed a named function -- an ACC->ACC sub-program -- into the holographic library.

        The function's body is assembled to a vector and bundled into ONE library vector under its
        name atom. A CALL to it later extracts it by name (unbind) and runs it on the current ACC.
        Functions are therefore data: composable, content-addressable, and stored in the same space
        as everything else. Define a function before assembling any program that CALLs it."""
        self.functions[name] = self.assemble(program)
        self.fn_atoms[name] = self._atom(f"fn:{name}", unitary=True)
        self.library = bundle([bind(self.fn_atoms[n], self.functions[n]) for n in self.functions])
        return self

    def assemble(self, program):
        """Encode a list of (opcode, operand) instructions as ONE program vector."""
        return bundle([bind(self.pos(i), self._instr(op, arg)) for i, (op, arg) in enumerate(program)])

    # ---- cleanup against the format's codebooks ----------------------------------------------
    @staticmethod
    def _nearest_loop(table, noisy):
        """The original cleanup: a Python loop of cosine calls, FIRST maximum wins (strict >). The recorded
        decision; the default. _nearest routes here unless fast_cleanup was opted into."""
        best, best_sim = None, -9.0
        for name, vec in table.items():
            s = cosine(noisy, vec)
            if s > best_sim:
                best_sim, best = s, name
        return best

    def _nearest(self, table, noisy):
        """Nearest-atom cleanup. Routes to the loop (default) or the opt-in SIMD matmul path (fast_cleanup=True).
        One router so every decode site -- run, run_batch, run_chunked, stack_pop -- honours the flag without a
        20-call-site sweep."""
        return self._nearest_fast(table, noisy) if self.fast_cleanup else self._nearest_loop(table, noisy)

    def _nearest_fast(self, table, noisy):
        """The SIMD form of _nearest: one (K,D)@(D,) matmul + argmax against a cached, row-normalised codebook
        matrix, instead of a Python loop of K cosine calls. The machine model's own advice ('numpy IS the vector
        unit -- do not reimplement it') applied to the VM's hottest inner loop. MEASURED: 148.5us -> 16.5us per
        decode (9.0x) on the 20-atom opcode+data codebook at dim 1024.

        TIE SEMANTICS: _nearest keeps the FIRST maximum (strict >); np.argmax also returns the first maximum, and
        cosine(noisy, v) == (v/|v|) . (noisy/|noisy|) up to float regrouping. Hammered with 3000 noisy decodes plus
        an exact two-atom tie: zero disagreements. Still OPT-IN (fast_cleanup=False default) under the QEM rule --
        a regrouped float path may flip a knife-edge tie somewhere we did not sample, and existing decisions must
        never flip silently. The selftest pins loop/matmul agreement so any future drift is caught loudly.

        The cache is keyed by the table OBJECT's id -- the format's codebooks are built once in __init__ and never
        mutated, so identity is a stable key; a new table (new id) just builds a new entry."""
        key = id(table)
        entry = self._cleanup_mats.get(key)
        if entry is None or entry[0] != len(table):
            names = list(table.keys())
            M = np.stack([table[n] for n in names])
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-300)   # guard an all-zero atom (cannot happen, cheap)
            entry = (len(table), names, Mn)
            self._cleanup_mats[key] = entry
        _, names, Mn = entry
        nv = float(np.linalg.norm(noisy))
        if nv == 0.0:
            return names[0]                                    # matches _nearest_loop: all cosines 0, first entry wins
        return names[int(np.argmax(Mn @ (noisy / nv)))]

    def decode_instruction(self, program_vec, i):
        """Read address i: return (opcode, operand) after cleanup. The honest, noisy read step.
        The operand is cleaned against the codebook the opcode implies -- function names for CALL,
        faculty names for APPLY, data atoms otherwise."""
        raw = unbind(program_vec, self.pos(i))
        op = self._nearest(self.op_atoms, unbind(raw, self.OP))
        if op in ("CALL", "ITERATE") and self.fn_atoms:
            arg = self._nearest(self.fn_atoms, unbind(raw, self.ARG))
        elif op == "APPLY":
            arg = self._nearest(self.fac_atoms, unbind(raw, self.ARG))
        elif op == "REPEAT":
            arg = self._nearest(self.cnt_atoms, unbind(raw, self.ARG))
        elif op in ("STORE", "RECALL"):
            arg = self._nearest(self.reg_atoms, unbind(raw, self.ARG))
        else:
            arg = self._nearest(self.data_atoms, unbind(raw, self.ARG))
        return op, arg

    def _read_addr(self, prog_spec, i, n):
        """Read instruction address i from a PRE-TRANSFORMED program spectrum. BIT-IDENTICAL to
        unbind(program_vec, pos(i)) -- same rfft(program_vec), same involution, same irfft -- but the program's
        spectrum is transformed ONCE per run() and reused across every instruction, instead of being recomputed
        on each one. This is Fill 1 (spectrum residency) applied to the hottest loop in the VM: the plan's
        observation that `run()` recomputes rfft(program_vec) on every opcode decode. Safe for the cleanup-gated
        decode precisely BECAUSE it is bit-exact -- it cannot flip a _nearest winner (the bind_batch discipline)."""
        from holographic.sampling_and_signal.holographic_fft import rfft as _rfft, irfft as _irfft
        return _irfft(prog_spec * _rfft(involution(self.pos(i))), n=n)

    # ---- executing a program -----------------------------------------------------------------
    def run(self, program_vec, init_acc=None, max_steps=512, _depth=0, handlers=None,
            stop=None, max_loop=64, converge_tol=0.999, branch_tol=0.5,
            init_regs=None, init_stack=None, return_state=False):
        """Execute the program vector; return (accumulator, trace_of_decoded_instructions).

        `init_acc` lets a program start from a given accumulator -- which is what makes a function an
        ACC->ACC transform that CALL can chain. A CALL instruction extracts the named function from the
        holographic library and runs it on the current ACC (with a recursion-depth guard). `handlers`
        maps a faculty name to a unary acc->acc function, supplied by the host (the mind); an APPLY
        whose faculty has no handler is a safe no-op (so the bare VM still runs the program).

        Control flow:
        * IFMATCH x : execute the NEXT instruction only if cosine(ACC, x) >= branch_tol, else skip it
          (a one-instruction conditional -- pair it with CALL for an if-then).
        * ITERATE f : re-apply library function f to ACC until it CONVERGES (cosine to the previous ACC
          >= converge_tol -- a fixed point), or the host `stop(acc)` predicate is satisfied (a desired
          OUTPUT is reached), or `max_loop` is hit. This is the input->process->feed-back-as-input loop
          that drives so much of the engine (cleanup, resonator, denoise), now expressible as a program.
          Its trace entry is the 4-tuple (op, f, iterations, reason) where reason is 'converged' /
          'goal' / 'maxloop'."""
        handlers = handlers or {}
        # RESIDENCY (Fill 1): transform the CONSTANT program vector ONCE and reuse its spectrum for every
        # instruction's address read, instead of recomputing rfft(program_vec) on each decode. Bit-identical.
        from holographic.sampling_and_signal.holographic_fft import rfft as _rfft
        prog_spec = _rfft(program_vec)
        n = program_vec.shape[0]
        acc = init_acc
        regs = dict(init_regs) if init_regs else {}  # seed from incoming state (chunk threading) or start fresh;
        stack = init_stack                           # exact carry -> no crosstalk added at a seam (ISA-4/5)
        trace = []
        pc = 0
        for _step in range(max_steps):              # cap on TOTAL instructions executed (the safety net)
            raw = self._read_addr(prog_spec, pc, n)          # residency: reuse the program's spectrum
            op = self._nearest(self.op_atoms, unbind(raw, self.OP))
            if op == "HALT":
                break
            if op == "CALL":
                fn = self._nearest(self.fn_atoms, unbind(raw, self.ARG))   # operand cleaned vs function names
                trace.append(("CALL", fn))
                if _depth < 8 and fn in self.functions:                    # guard against runaway recursion
                    sub = unbind(self.library, self.fn_atoms[fn])          # pull the function body out of the library
                    acc, _ = self.run(sub, init_acc=acc, _depth=_depth + 1, handlers=handlers,
                                      stop=stop, max_loop=max_loop, converge_tol=converge_tol, branch_tol=branch_tol)
                pc += 1
                continue
            if op == "APPLY":
                fac = self._nearest(self.fac_atoms, unbind(raw, self.ARG))  # operand cleaned vs faculty names
                trace.append(("APPLY", fac))
                if acc is not None and fac in handlers:                     # the host runs the faculty on ACC
                    acc = handlers[fac](acc)
                pc += 1
                continue
            if op == "ITERATE":                                            # fixed-point loop over a library function
                fn = self._nearest(self.fn_atoms, unbind(raw, self.ARG))
                iters, reason = 0, "maxloop"
                if _depth < 8 and fn in self.functions and acc is not None:
                    body = unbind(self.library, self.fn_atoms[fn])
                    for iters in range(1, max_loop + 1):
                        prev = acc
                        acc, _ = self.run(body, init_acc=acc, _depth=_depth + 1, handlers=handlers,
                                          stop=stop, max_loop=max_loop, converge_tol=converge_tol, branch_tol=branch_tol)
                        if stop is not None and stop(acc):                  # the desired OUTPUT was reached
                            reason = "goal"
                            break
                        if cosine(acc, prev) >= converge_tol:               # ACC stopped changing: a fixed point
                            reason = "converged"
                            break
                trace.append(("ITERATE", fn, iters, reason))
                pc += 1
                continue
            if op == "REPEAT":                                         # counted loop: run the next CALL n times
                cnt = self._nearest(self.cnt_atoms, unbind(raw, self.ARG))
                trace.append(("REPEAT", cnt))
                nraw = self._read_addr(prog_spec, pc + 1, n)             # the instruction to repeat (expects a CALL)
                if self._nearest(self.op_atoms, unbind(nraw, self.OP)) == "CALL":
                    fn = self._nearest(self.fn_atoms, unbind(nraw, self.ARG))
                    trace.append(("CALL", fn))
                    if _depth < 8 and fn in self.functions:
                        body = unbind(self.library, self.fn_atoms[fn])
                        for _ in range(max(1, cnt)):
                            acc, _ = self.run(body, init_acc=acc, _depth=_depth + 1, handlers=handlers,
                                              stop=stop, max_loop=max_loop, converge_tol=converge_tol, branch_tol=branch_tol)
                    pc += 2                                            # skip REPEAT and the CALL it consumed
                else:
                    pc += 1                                            # not a CALL -> REPEAT is a no-op
                continue
            if op == "IFMATCH":                                            # conditional: gate the NEXT instruction
                tgt = self._nearest(self.data_atoms, unbind(raw, self.ARG))
                matched = acc is not None and cosine(acc, self.data_atoms[tgt]) >= branch_tol
                trace.append(("IFMATCH", tgt))
                pc += 1 if matched else 2                                   # skip the guarded instruction on no-match
                continue
            if op == "STORE":                                              # ACC -> register slot (exact)
                reg = self._nearest(self.reg_atoms, unbind(raw, self.ARG))
                regs[reg] = acc                                            # separate named slot: no crosstalk
                trace.append(("STORE", reg))
                pc += 1
                continue
            if op == "RECALL":                                             # register slot -> ACC (exact)
                reg = self._nearest(self.reg_atoms, unbind(raw, self.ARG))
                if reg in regs:
                    acc = regs[reg]                                        # exact read -- the value is returned verbatim
                trace.append(("RECALL", reg))
                pc += 1
                continue
            if op == "PUSH":                                               # push ACC onto the permute-stack
                if acc is not None:
                    stack = stack_push(stack, acc)
                trace.append(("PUSH",))
                pc += 1
                continue
            if op == "POP":                                                # pop the top of the permute-stack into ACC
                if stack is not None:
                    acc, stack = stack_pop(stack, list(self.data_atoms.values()))  # cleaned vs the value codebook
                trace.append(("POP",))
                pc += 1
                continue
            arg = self._nearest(self.data_atoms, unbind(raw, self.ARG))
            trace.append((op, arg))
            d = self.data_atoms[arg]
            if op == "LOAD" or acc is None:        # guard: a value-op before any LOAD acts as LOAD,
                acc = d                            # so a corrupted program can't crash the interpreter
            elif op == "BIND":
                acc = bind(acc, d)
            elif op == "BUNDLE":
                acc = bundle([acc, d])
            elif op == "PERMUTE":
                acc = permute(acc, 1)
            pc += 1
        if return_state:                             # chunk threading wants the register file + stack carried out
            return acc, trace, regs, stack
        return acc, trace

    def run_batch(self, program_vec, init_accs, max_steps=512):
        """Run ONE straight-line program over a BATCH of accumulators (N, D) in a SINGLE interpret pass -- the
        data-parallel ('as below') form of run(). The architecture sweep found the hot spot: the VM decodes each
        instruction once (an unbind + a nearest-atom lookup -- the expensive per-instruction work), then applies
        the value op to the accumulator; running the SAME program over N data items therefore meant N full Python
        interpret passes, re-decoding every instruction N times. Here the decode happens ONCE and the value op hits
        all N rows at once with the batch-aware primitives (bind_batch, roll axis=-1, broadcast-bundle). N items
        cost one decode loop, not N -- the amortisation measured in the tests.

        Why the plain ops couldn't already do this: bind() hardcodes n=a.shape[0] and permute() uses np.roll with
        no axis, so both silently corrupt a 2-D batch -- they were written 1-D-only. bind_batch and roll(axis=-1)
        are the batch-correct forms (matching the scalar ops to machine epsilon), which is what this uses.

        SCOPE (kept honest): straight-line VALUE + REGISTER programs (LOAD/BIND/BUNDLE/PERMUTE/STORE/RECALL/HALT).
        Data-dependent control (IFMATCH) and host/library ops (CALL/ITERATE/REPEAT/APPLY/PUSH/POP) are NOT
        batchable -- their control flow or per-item cleanup diverges across the batch -- so they raise a clear
        error naming the op rather than return a silently-wrong result; loop run() per item for those. Matches the
        scalar run() to ~1e-12 (the bind_batch tie already on record), so tie-sensitive paths keep run(). Returns
        the final (N, D) accumulator batch."""
        acc = np.asarray(init_accs, float)
        if acc.ndim != 2:
            raise ValueError("run_batch expects init_accs of shape (N, D)")
        N, D = acc.shape
        # RESIDENCY (Fill 1): transform the constant program once, reuse across the decode loop. Bit-identical.
        from holographic.sampling_and_signal.holographic_fft import rfft as _rfft
        prog_spec = _rfft(program_vec)
        n = program_vec.shape[0]
        regs = {}
        _unbatchable = {"CALL", "ITERATE", "REPEAT", "APPLY", "IFMATCH", "PUSH", "POP"}
        pc = 0
        for _step in range(max_steps):
            raw = self._read_addr(prog_spec, pc, n)
            op = self._nearest(self.op_atoms, unbind(raw, self.OP))        # decoded ONCE for the whole batch
            if op == "HALT":
                break
            if op in _unbatchable:
                raise ValueError(f"run_batch does not support the control/host op {op!r} -- straight-line "
                                 f"value+register programs only; loop run() per item for control-flow programs")
            if op == "STORE":
                regs[self._nearest(self.reg_atoms, unbind(raw, self.ARG))] = acc
                pc += 1; continue
            if op == "RECALL":
                reg = self._nearest(self.reg_atoms, unbind(raw, self.ARG))
                if reg in regs:
                    acc = regs[reg]
                pc += 1; continue
            arg = self._nearest(self.data_atoms, unbind(raw, self.ARG))
            d = self.data_atoms[arg]
            if op == "LOAD":
                acc = np.broadcast_to(d, (N, D)).copy()                   # a program constant -> same for all rows
            elif op == "BIND":
                acc = bind_batch(acc, np.broadcast_to(d, (N, D)))         # the batch-correct bind (axis=-1)
            elif op == "BUNDLE":
                total = acc + d
                norm = np.linalg.norm(total, axis=1, keepdims=True)
                acc = np.where(norm > 0, total / norm, total)
            elif op == "PERMUTE":
                acc = np.roll(acc, 1, axis=-1)                            # per-row cyclic shift
            pc += 1
        return acc

    # ---- running a program too long for one structure (the chunk_route lesson, applied to instructions) ----
    # Control ops that GATE or CONSUME the instruction after them -- splitting a chunk between them and their
    # target would break the construct, so a chunk is never allowed to END on one of these.
    _SPANS_NEXT = ("IFMATCH", "REPEAT")

    def state_rows(self, acc, regs=None, stack=None):
        """The machine state as an (n_slots, D) array of ROWS -- ACC, then each register R0..R7, then the stack --
        zeros for an empty slot. Unlike state_to_vector (one bundled, lossy vector), this keeps each slot as a
        SEPARATE ROW, so a SEQUENCE of states delta-compresses well in a DeltaChain (a register that didn't change
        between seams is an unchanged row, costing nothing). The right shape for an execution replay log."""
        regs = regs or {}
        rows = [acc if acc is not None else np.zeros(self.dim)]
        for r in REGISTERS:
            v = regs.get(r)
            rows.append(v if v is not None else np.zeros(self.dim))
        rows.append(stack if stack is not None else np.zeros(self.dim))
        return np.array(rows, float)

    def run_chunked(self, program, chunk=14, init_acc=None, handlers=None,
                    stop=None, max_loop=64, converge_tol=0.999, branch_tol=0.5, record=False):
        """Run a program TOO LONG for one structure, by splitting it into <=chunk-instruction pieces -- each
        its OWN clean program vector -- and THREADING the accumulator across them. This is the way past the
        single-program capacity cap (~20-32 instructions at dim 1024), and it is the chunk_route lesson applied
        to instructions: don't overstuff one structure, keep each piece inside its capacity and carry the state
        across the seam.

        Why not just CALL sub-programs: CALL pulls each function out of a BUNDLED library, and bundling several
        function-vectors into one library re-introduces the very cliff we're escaping (measured: 60 BIND ops via
        CALL collapse to cosine 0.06, but host-threaded chunks run them at cosine 1.000). So the chunks are
        independent program vectors, not library entries -- the accumulator is the only thing that crosses a
        boundary, exactly as a re-anchored route carries only its last clean tile.

        Scope (kept honest): this threads the ACCUMULATOR -- the data-flow of LOAD/BIND/BUNDLE/PERMUTE and the
        per-chunk effects of APPLY/CALL. A control construct that gates or consumes its next instruction
        (`IFMATCH x; <gated>`, `REPEAT n; <CALL>`) is kept INTACT within a chunk (a chunk never ends on one),
        but a single construct must not be relied on to span a boundary beyond that. Put HALT at the end (a
        trailing HALT is stripped; per-chunk HALTs are added); a HALT in the MIDDLE stops the whole run, like
        the flat interpreter. `chunk` must stay WELL UNDER the dim's reliable program length, with margin: at
        dim 1024 the decode is solid through ~18 instructions but turns OPERAND-DEPENDENT right at the ~20 edge
        (measured: a 20-instruction chunk decodes for some operand sequences and fails for others), so the
        default 14 leaves a deliberate margin. The reliable length grows with dimension, so raise `chunk` at
        higher dim (a chunk of 20 is solid at dim 2048+). Returns (accumulator, trace) just like run(), with
        trace the chunks' traces concatenated."""
        instrs = list(program)
        if instrs and instrs[-1][0] == "HALT":               # we add per-chunk HALTs; drop a trailing one
            instrs = instrs[:-1]
        acc, regs, stack, full_trace = init_acc, {}, None, []
        seam_states = []                                     # per-seam state rows, for a replay log (record=True)
        i, n = 0, len(instrs)
        while i < n:
            end = min(i + chunk, n)
            while end < n and instrs[end - 1][0] in self._SPANS_NEXT:
                end += 1                                     # never split a gate/repeat from the instruction it targets
            seg = instrs[i:end] + [("HALT", "")]             # one clean, independent program vector per chunk
            # carry the FULL machine state across the seam: accumulator AND the register file AND the stack,
            # each in its EXACT representation (a dict / the stack vector), so STORE in one chunk is readable by
            # RECALL in a later one and crosstalk never accumulates at a boundary (the exact-carry choice -- see
            # state_to_vector for the composable bundled alternative and why it is NOT used per-seam).
            acc, tr, regs, stack = self.run(self.assemble(seg), init_acc=acc, handlers=handlers,
                                            init_regs=regs, init_stack=stack, return_state=True, stop=stop,
                                            max_loop=max_loop, converge_tol=converge_tol, branch_tol=branch_tol)
            full_trace += tr
            if record:
                seam_states.append(self.state_rows(acc, regs, stack))   # snapshot the state at this seam
            if len(tr) < len(seg) - 1:                       # a HALT (or stop) fired mid-chunk -> stop the whole run
                break
            i = end
        if record:
            return acc, full_trace, seam_states
        return acc, full_trace

    # ---- the machine state AS A COMPOSABLE VECTOR (a continuation) -- the VSA-native win ------------------
    def state_to_vector(self, acc, regs=None, stack=None):
        """Bundle the whole machine STATE -- accumulator + register file + stack -- into ONE composable
        hypervector (a 'continuation'): bind each part to its role and superpose. This is the VSA-native payoff
        the chunk-threading sets up: a paused computation becomes a first-class VALUE you can STORE in memory,
        recall, compose with other states, or resume later -- the same 'a program/state is just a vector'
        composability the inception layer already uses for programs.

        HONEST COST (measured in the tests): a bundle has finite capacity, so reading a part back is a NOISY
        unbind that must be CLEANED against a codebook -- exact-after-cleanup for cleanup-able (atom-valued)
        slots, lossy for arbitrary continuous values, and the fidelity falls as more slots are packed (the
        capacity cliff, ~1/sqrt(#slots)). This is WHY run_chunked threads the EXACT register dict across each
        seam instead of bundling: bundling per-seam would compound that crosstalk over a long program. So the
        bundled state is for SNAPSHOT / store / compose / resume -- where carrying one vector is the point -- and
        the exact dict is for the hot per-seam carry. VSA-native where it is beneficial, exact where it is."""
        parts = []
        if acc is not None:
            parts.append(bind(self.ACC, acc))
        for r, v in (regs or {}).items():
            if v is not None:
                parts.append(bind(self.reg_atoms[r], v))     # each register under its own name role
        if stack is not None:
            parts.append(bind(self.STK, stack))
        return bundle(parts) if parts else np.zeros(self.dim)

    def state_from_vector(self, vec, reg_names=(), codebook=None):
        """Inverse of state_to_vector: unbind each role and CLEAN against `codebook` (a list of candidate value
        vectors, e.g. the data atoms) to recover (acc, regs, stack). With no codebook the raw (noisy) unbinds are
        returned for the caller to clean. The stack is returned as its raw bundle for stack_pop to clean on the
        way out. Atom-valued slots come back exact after cleanup; see the tests for the measured fidelity."""
        def _read(role):
            raw = unbind(vec, role)
            if not codebook:
                return raw
            return max(codebook, key=lambda c: cosine(raw, c))   # snap to the nearest known value (cleanup)
        acc = _read(self.ACC)
        regs = {r: _read(self.reg_atoms[r]) for r in reg_names}
        stack = unbind(vec, self.STK)
        return acc, regs, stack

    # ---- nesting (the inception layer): a program is just another value to store --------------
    def as_file(self, content_vec):
        """Wrap a vector as a 'file' under the SLOT role, ready to drop onto a disk."""
        return bind(self.SLOT, content_vec)

    def disk(self, content_vec, other_files=()):
        """A 'disk': the SLOT-file holding `content_vec`, bundled with any other files on the disk.
        More files per disk => more crosstalk => a buried program corrupts at a shallower depth."""
        return bundle([self.as_file(content_vec), *other_files])

    def open_slot(self, disk_vec):
        """Recover the SLOT-file's contents from a disk (noisy if the disk holds other files)."""
        return unbind(disk_vec, self.SLOT)

    def junk_files(self, n, tag):
        """n deterministic distractor files, to simulate a disk that holds other things too."""
        return [bind(self._atom(f"f:{tag}:{j}", unitary=True), self._atom(f"j:{tag}:{j}"))
                for j in range(n)]
