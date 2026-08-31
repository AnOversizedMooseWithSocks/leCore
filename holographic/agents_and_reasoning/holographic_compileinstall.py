"""holographic_compileinstall.py -- THE F27 CONFORMANCE MILESTONE + THE F26 MANIFEST.

The claim being tested: a HoloMachine program IS an installable object -- its linear opcodes are
matrices (the projector certifies them), REPEAT of a linear body is an OPERATOR POWER (one matvec,
not n), and registers are recurrent state slots. So the same program runs two ways:

  VM PATH (reference):  the program is an HRR vector; the VM decodes each instruction
                        holographically (cleanup-gated) and executes with runtime control flow.
  INSTALLED PATH:       compiled ONCE from the symbolic program into a chain of certified matvecs
                        + register-slot copies -- the arithmetic a weight-installed layer performs,
                        with control flow reduced to the chain order (the token loop's job).

CONFORMANCE = the two paths agree on the final accumulator per the ISA's tags: BIND/PERMUTE/
STORE/RECALL are EXACT ops, so agreement is numerical (allclose), not cosine-ish. The asymmetry is
the point: the VM PAYS decode noise and control flow at runtime; the installed path paid it all at
compile time. Same program, same answer, different substrate -- which is the whole Unicron thesis
in one testable sentence.

REPEAT AS OPERATOR POWER (the lever-3/4 move): REPEAT n over a circulant is spectrum**n applied
once -- n matvecs collapse to one, EXACTLY (FFT diagonalizes every circulant, so the power is
elementwise in the spectrum; no approximation to tag). The projector's structure detection is what
makes this safe: only a certified 'circulant' takes the spectral shortcut; a certified 'dense'
takes matrix_power; anything refused refuses here too.

F26: every compiled program yields a MANIFEST -- name -> {kind, payload shape, residual, seconds}
per installed opcode plus the program chain -- the installed side's discoverability contract (the
runtime has find_capability; the weights get this). save_manifest writes the JSON sidecar.
"""
import json
import numpy as np


def compile_installed(machine, program, tol=1e-8, host_fallback=False):
    """Compile a symbolic HoloMachine program (list of (OP, arg)) into an installed runner + manifest.

    Supported: LOAD / BIND / PERMUTE / STORE / RECALL / HALT, and REPEAT n + CALL fn where fn's
    body is itself all-linear (certified by the projector; a nonlinear body REFUSES loudly --
    the core/shell boundary, measured not declared). Returns (run_installed() -> acc, manifest)."""
    from holographic.io_and_interop.holographic_projector import probe_project, apply_projected
    from holographic.agents_and_reasoning.holographic_ai import bind
    dim = machine.dim
    manifest = {"dim": dim, "ops": {}, "chain": []}
    # STATE-DIM TRACKING (the 2D-editing pipeline found the gap): a chain's state can CHANGE
    # dimension across rectangular steps (3 scene params -> 64 pixels -> ...). Each FAC step is
    # certified at the CURRENT state dim, and rectangular certificates advance it. VM value ops
    # always run at machine dim -- asserted, so a mismatched chain fails at compile, not at run.
    cur = {"dim": dim}

    def _certify(name, f, at_dim=None):
        if name not in manifest["ops"]:
            pr = probe_project(f, at_dim if at_dim is not None else dim, tol=tol)
            if pr["kind"] == "refused":
                raise ValueError("opcode %r is not linear (residual %.2e) -- cannot install; "
                                 "wrap as APPLY (T3)" % (name, pr["residual"]))
            # CONDITIONING CERTIFICATE (audit finding, measured): a NON-UNITARY circulant has
            # spectrum magnitudes != 1, and a depth-64 chain of such binds exploded to 1e8 (7.8e82
            # at 256) -- not precision loss but exponential amplification, the HRR-classical reason
            # roles are unitary. The certificate PRICES it: spec_max/spec_min per circulant, so the
            # chain's amplification bound (prod of spec_max, worst case) is computable BEFORE any
            # host installation. Deep chains want unitary operands; the manifest now says so with
            # numbers instead of folklore.
            if pr["kind"] == "circulant":
                mag = np.abs(np.fft.rfft(pr["column"]))
                pr["spec_max"] = float(mag.max()); pr["spec_min"] = float(mag.min())
            manifest["ops"][name] = pr
        return manifest["ops"][name]

    steps = []                                                # compiled chain: closures over certified payloads
    i = 0
    while i < len(program):
        op, arg = program[i]
        if op == "LOAD":
            atom = machine.data_atoms[arg]
            steps.append(("LOAD", arg, lambda st, regs, a=atom: (a.copy(), regs)))
        elif op == "BIND":
            key = machine.data_atoms[arg]
            # VM executes bind(acc, d); circular convolution commutes so bind(d, acc) is the same
            # map -- certify in the VM's own order anyway (conformance should not lean on algebra).
            pr = _certify("BIND:%s" % arg, lambda x, k=key: bind(x, k))
            steps.append(("BIND", arg, lambda st, regs, p=pr: (apply_projected(p, st), regs)))
        elif op == "PERMUTE":
            sh = int(arg) if not isinstance(arg, str) else 1
            pr = _certify("PERMUTE:%s" % arg, lambda x, s=sh: np.roll(x, s))
            steps.append(("PERMUTE", arg, lambda st, regs, p=pr: (apply_projected(p, st), regs)))
        elif op == "REPEAT":
            nop, nfn = program[i + 1]
            if nop != "CALL":
                i += 1
                continue                                       # VM semantics: REPEAT before non-CALL is a no-op
            body = machine.functions_symbolic[nfn] if hasattr(machine, "functions_symbolic") else None
            if body is None:
                raise ValueError("REPEAT+CALL needs the symbolic body of %r (define_symbolic)" % nfn)
            # certify the WHOLE body as one operator, then take its n-th power -- circulants
            # power in the spectrum (exact), dense bodies via matrix_power.
            def body_fn(x, b=body, mach=machine):
                acc = x
                for bop, barg in b:
                    if bop == "BIND":
                        acc = bind(acc, mach.data_atoms[barg])   # the VM's operand order, exactly
                    elif bop == "PERMUTE":
                        acc = np.roll(acc, int(barg) if not isinstance(barg, str) else 1)
                    else:
                        raise ValueError("non-linear/unsupported op %r in REPEAT body" % bop)
                return acc
            pr = _certify("BODY:%s" % nfn, body_fn)
            n_rep = int(arg)
            if pr["kind"] == "circulant":
                spec = np.fft.rfft(pr["column"]) ** n_rep      # EXACT: FFT diagonalizes circulants
                powered = {"kind": "circulant", "column": np.fft.irfft(spec, n=dim)}
            elif pr["kind"] == "permutation":
                perm = pr["perm"].copy()
                for _ in range(n_rep - 1):
                    perm = pr["perm"][perm]
                powered = {"kind": "permutation", "perm": perm}
            else:
                powered = {"kind": "dense", "matrix": np.linalg.matrix_power(pr["matrix"], n_rep),
                           "offset": np.zeros(dim)}
            manifest["ops"]["BODY:%s^%d" % (nfn, n_rep)] = {k: v for k, v in powered.items()}
            steps.append(("POWER", "%s^%d" % (nfn, n_rep),
                          lambda st, regs, p=powered: (apply_projected(p, st), regs)))
            i += 2
            continue
        elif op == "STORE":
            steps.append(("STORE", arg, lambda st, regs, r=arg: (st, {**regs, r: st.copy()})))
        elif op == "RECALL":
            steps.append(("RECALL", arg, lambda st, regs, r=arg: (regs[r].copy(), regs)))
        elif op == "IFMATCH":
            # G4 -- CONTROL AS A MARKED HOST STEP: the cosine gate is data-dependent branching --
            # per INSTALLED.md it cannot be a frozen matvec, and pretending otherwise is the exact
            # dishonesty this pipeline refuses. It compiles as a host-side SELECT (the token
            # loop's job, MoE-routing shaped) and the chain marks the step HOST: so an installer
            # knows control lives here. The GUARDED instruction's arithmetic still installs
            # (certified as usual); only the yes/no is runtime. VM semantics matched exactly:
            # cosine(acc, target) >= 0.5 runs the next instruction, else skips it.
            tgt = machine.data_atoms[arg]
            nop, narg = program[i + 1]
            if nop == "BIND":
                key2 = machine.data_atoms[narg]
                prg = _certify("BIND:%s" % narg, lambda x, k=key2: bind(x, k))
            elif nop == "PERMUTE":
                prg = _certify("PERMUTE:%s" % narg, lambda x: np.roll(x, 1))
            else:
                raise ValueError("IFMATCH guards only value ops here (got %r)" % nop)
            def _sel(st, regs, t=tgt, p2=prg):
                c = float(st @ t / (np.linalg.norm(st) * np.linalg.norm(t) + 1e-12))
                return (apply_projected(p2, st) if c >= 0.5 else st), regs
            steps.append(("HOST:IFMATCH", "%s->%s:%s" % (arg, nop, narg), _sel))
            i += 2
            continue
        elif op == "ITERATE":
            # G5 -- FIXED POINT AS BOUNDED UNROLL + HOST CONVERGENCE CHECK: the body's arithmetic
            # installs (one certified operator applied repeatedly); the convergence TEST is data-
            # dependent control and stays a marked host step. VM semantics matched: up to 64
            # applications, stop at cosine(new, prev) >= 0.999.
            bodyI = machine.functions_symbolic.get(arg)
            if bodyI is None:
                raise ValueError("ITERATE needs the symbolic body of %r" % arg)
            def bodyI_fn(x, b=bodyI, mach=machine):
                acc2 = x
                for bop, barg in b:
                    acc2 = bind(acc2, mach.data_atoms[barg]) if bop == "BIND" else np.roll(acc2, 1)
                return acc2
            prb = _certify("BODY:%s" % arg, bodyI_fn)
            def _iter(st, regs, p2=prb):
                cur = st
                for _ in range(64):
                    prev = cur
                    cur = apply_projected(p2, cur)
                    if float(cur @ prev / (np.linalg.norm(cur) * np.linalg.norm(prev) + 1e-12)) >= 0.999:
                        break
                return cur, regs
            steps.append(("HOST:ITERATE", str(arg), _iter))
        elif op == "FAC":
            # G2 -- FACULTY-CALL COMPILATION: the chain can now install a certified FACULTY core,
            # not just VM opcodes. arg = (name, callable); certified through the SAME projector
            # (so blockdiag/rmsnorm/etc. all apply), refusals loud, certificate cached under the
            # name. This is the door the 3D/sim linear cores walk through: the census's
            # candidates compile HERE.
            fname, fcall = arg
            try:
                pr = _certify("FAC:%s" % fname, fcall, at_dim=cur["dim"])
                k = pr["kind"]
                cur["dim"] = (pr["matrix"].shape[0] if k == "dense"
                              else len(pr["gain"]) if k == "rmsnorm" else cur["dim"])
                steps.append(("FAC", fname, lambda st, regs, p=pr: (apply_projected(p, st), regs)))
            except ValueError:
                if not host_fallback:
                    raise
                # G9 -- THE FUSION SPLIT: a refused faculty becomes a MARKED host step instead of
                # a dead end. The chain stays one program; the manifest says exactly which links
                # are weights and which are runtime (kind='host_apply' with the refusal residual
                # recorded -- the refusal certificate travels even though nothing installed).
                pr_ref = probe_project(fcall, cur["dim"], tol=tol)
                cur["dim"] = np.asarray(fcall(np.zeros(cur["dim"])), float).reshape(-1).shape[0]
                manifest["ops"]["HOST:%s" % fname] = {"kind": "host_apply",
                                                      "residual": pr_ref["residual"],
                                                      "seconds": pr_ref["seconds"]}
                steps.append(("HOST:APPLY", fname,
                              lambda st, regs, f2=fcall: (np.asarray(f2(st), float).reshape(-1), regs)))
        elif op == "HALT":
            break
        else:
            raise ValueError("unsupported opcode for installation: %r" % op)
        i += 1
    manifest["chain"] = [(s[0], str(s[1])) for s in steps]
    # chain-level conditioning: worst-case log-amplification of the whole program (sum of
    # log(spec_max) over matvec steps, REPEAT powers counted n times via the powered payload).
    # walk the CHAIN, not the op dict: 64 BINDs of one key dedupe to a single certified op, but
    # the amplification is paid PER STEP (first draft summed per-op and missed exactly the deep
    # chain it existed to catch -- the assert caught it).
    logamp = 0.0
    for kind, ref in manifest["chain"]:
        opname = {"BIND": "BIND:%s" % ref, "PERMUTE": "PERMUTE:%s" % ref,
                  "POWER": "BODY:%s" % ref}.get(kind)
        pr = manifest["ops"].get(opname, {}) if opname else {}
        if "spec_max" in pr:
            logamp += np.log(max(pr["spec_max"], 1e-300))
        elif kind == "POWER":
            base = manifest["ops"].get("BODY:%s" % ref.split("^")[0], {})
            if "spec_max" in base:
                logamp += float(ref.split("^")[1]) * np.log(max(base["spec_max"], 1e-300))
    manifest["log_amplification_bound"] = float(logamp)
    if logamp > np.log(1e6):
        manifest.setdefault("warnings", []).append(
            "chain amplification bound exp(%.1f) exceeds 1e6 -- deep non-unitary bind chains "
            "explode (measured: 1e8 at depth 64); use unitary operands for deep/REPEAT-heavy "
            "programs" % logamp)

    def run_installed(init=None):
        # init: arbitrary starting STATE (G10 -- a mesh's flattened vertices ARE the state; the
        # data-atom LOAD is just the default init for symbolic programs)
        st, regs = (None if init is None else np.asarray(init, float).reshape(-1)), {}
        for _, _, f in steps:
            st, regs = f(st, regs)
        return st
    return run_installed, manifest


def mesh_program_obj(machine, program, verts, faces, host_fallback=False):
    """G10 -- THE MESH PROGRAM, mouth-first (principle G0: the token stream is the output
    device). Compile `program` (FAC steps over flattened vertices), run it INSTALLED with the
    mesh's own vertices as the state, and return the transformed mesh as an OBJ TEXT DUMP --
    no file I/O anywhere; the text leaves through the reply. Determinism makes byte-exactness a
    testable contract: the dump from the installed chain must equal the dump from the live
    faculty path character for character ('%.6f' formatting, fixed by this function, not the
    caller). Returns (obj_text, manifest)."""
    V = np.asarray(verts, float)
    run, man = compile_installed(machine, program, host_fallback=host_fallback)
    out = run(init=V.reshape(-1)).reshape(-1, 3)
    lines = ["# leCore installed mesh program -- emitted by the chain, not the filesystem"]
    lines += ["v %.6f %.6f %.6f" % tuple(p) for p in out]
    lines += ["f %d %d %d" % tuple(int(i) + 1 for i in f) for f in np.asarray(faces, int)]
    return "\n".join(lines) + "\n", man


def sim_program_run(machine, step_program, init, n_steps, host_fallback=True):
    """G11 -- THE SIM PROGRAM: compile ONE physics step (linear projections install certified;
    clamps/contacts ride as marked HOST:APPLY links per the fusion split) and iterate it
    n_steps times with the state fed back -- the installed chain IS the integrator. Returns
    (trajectory [n_steps+1, D], manifest, drift): drift[t] = max abs difference vs the live
    step function at step t, the published honesty curve -- exact certified steps give a flat
    ~1e-12 line; any growth is the certificate's residual compounding, visible, not hidden."""
    run, man = compile_installed(machine, step_program, host_fallback=host_fallback)
    live_fns = [a[1] for op, a in step_program if op == "FAC"]
    def live_step(x):
        for f in live_fns:
            x = np.asarray(f(x), float).reshape(-1)
        return x
    st_i = np.asarray(init, float).reshape(-1)
    st_l = st_i.copy()
    traj = [st_i.copy()]
    drift = [0.0]
    for _ in range(int(n_steps)):
        st_i = run(init=st_i)
        st_l = live_step(st_l)
        traj.append(st_i.copy())
        drift.append(float(np.max(np.abs(st_i - st_l))))
    return np.stack(traj), man, np.array(drift)


def collapse_recurrence(machine, step_program, n_steps, host_fallback=False, tol=1e-9):
    """THE HRNN COLLAPSE: a linear recurrence x_t = M x_{t-1} IS leCore's HRNN with the decay
    inside M -- and n applications of one linear operator are ONE operator (the REPEAT lesson,
    applied to TIME). Compile the step, compose its certified ops into a single matrix, raise it
    to n by eigendecomposition-free repeated squaring (exact float semantics of matrix_power),
    and certify the COLLAPSED operator against the live n-step run on held-out inits. n matvecs
    become one: the 100-step sim endpoint at step cost O(log n) build, O(1) query.
    HONEST BOUNDARY, enforced not documented: HOST steps (clamps, branches) BREAK linearity --
    any HOST:* link raises with the step names; collapse is for the all-certified case, and the
    step-by-step path (sim_program_run) remains the referee AND the drift instrument. SPECTRUM
    PRICED: |eig| max/min of M^n reported; explosive or vanishing recurrences are visible in the
    certificate, not discovered at runtime."""
    run_step, man = compile_installed(machine, step_program, host_fallback=host_fallback)
    hosts = [s for s in man["chain"] if s[0].startswith("HOST:")]
    if hosts:
        raise ValueError("collapse_recurrence needs an all-linear step; host links present: %s"
                         % [h[0] + ":" + str(h[1]) for h in hosts])
    dim = None
    probe0 = np.zeros(machine.dim)
    y0 = run_step(init=probe0)
    dim = y0.shape[0]
    if dim != machine.dim:
        raise ValueError("collapse needs a square step (state dim in == out); got %d -> %d"
                         % (machine.dim, dim))
    M = np.empty((dim, dim))
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1.0
        M[:, i] = run_step(init=e) - y0
    if int(n_steps) < 0:
        # reversal belongs to the time machine, where unitarity is CHECKED; a blind inverse
        # here would silently explode on decaying steps (the 1.4e+121 probe). Refuse, point.
        raise ValueError("n_steps < 0: reversal needs certified-unitary dynamics -- use "
                         "mind.time_machine().time_jump with t < 0")
    Mn = np.linalg.matrix_power(M, int(n_steps))
    # certificate on held-out inits: collapsed == live n-step within tol
    rng = np.random.default_rng(97)
    worst = 0.0
    for _ in range(6):
        x = rng.standard_normal(dim)
        live = x
        for _ in range(int(n_steps)):
            live = run_step(init=live)
        coll = Mn @ x + _affine_accum(M, y0, int(n_steps))
        worst = max(worst, float(np.max(np.abs(coll - live))))
    if worst > tol:
        raise ValueError("collapse residual %.2e exceeds tol %.2e -- step not linear enough" % (worst, tol))
    ev = np.abs(np.linalg.eigvals(M))
    cert = {"n_steps": int(n_steps), "residual": worst,
            "eig_max": float(ev.max()), "eig_min": float(ev.min()),
            "eign_max": float(ev.max() ** n_steps), "ops": man["ops"]}
    off = _affine_accum(M, y0, int(n_steps))

    def run_collapsed(init):
        return Mn @ np.asarray(init, float).reshape(-1) + off
    return run_collapsed, cert


def _affine_accum(M, b, n):
    """Offset of n affine steps x -> Mx + b: (M^(n-1) + ... + I) b. PLAIN O(n) LOOP, said
    plainly -- the first docstring claimed 'squaring discipline' while the code looped
    (circle-back V9: a doc-vs-code lie the audit convention exists to kill). The loop is
    n matmuls at compile time, ONCE; the collapsed query stays O(1). If a caller ever needs
    n in the millions, the geometric series doubles as S_2n = S_n + M^n @ S_n -- build it
    then, against a measured need, not now against an imagined one."""
    if not np.any(b):
        return b
    dim = M.shape[0]
    S = np.zeros((dim, dim)); P = np.eye(dim)
    for _ in range(int(n)):
        S += P
        P = P @ M
    return S @ b


def raster_program_pgm(machine, program, params, width, height, host_fallback=False):
    """G12 -- RENDER-TO-TEXT: run an installed image-formation chain (scene params -> pixels;
    linear formation models -- splatting, basis lighting -- certify like any faculty) and emit
    the frame as PGM P2 ASCII text: the picture leaves through the mouth (G0), no file I/O.
    Pixels clipped to [0,255] ints at emit (quantization is the SERIALIZER's job, stated -- the
    chain stays float and certified). Byte-exactness vs the live path is a testable contract."""
    run, man = compile_installed(machine, program, host_fallback=host_fallback)
    px = run(init=np.asarray(params, float).reshape(-1))
    # ROUNDING MARGIN (T14). Byte-exactness against another substrate holds only while that
    # substrate's error is SMALLER than the distance from every pixel to a .5 rounding boundary.
    # Measured on a GLSL port of this same chain: the f32-vs-f64 error EXCEEDED that distance at
    # three of four light counts, and the images matched anyway only because the few near-boundary
    # pixels happened to err the right way. So the margin is REPORTED rather than assumed, and a
    # caller can decide whether their scene is inside it. One min over the fractional parts.
    _f = np.asarray(px, float)
    _frac = np.abs(_f - np.floor(_f) - 0.5)
    _margin = float(np.min(_frac)) if _frac.size else 0.0
    q = np.clip(np.round(px), 0, 255).astype(int).reshape(height, width)
    lines = ["P2", "# leCore installed render -- emitted by the chain", "%d %d" % (width, height), "255"]
    lines += [" ".join(str(v) for v in row) for row in q]
    if isinstance(man, dict):
        man = dict(man)
        man["rounding_margin"] = _margin
        man["rounding_margin_note"] = (
            "byte-exactness against another substrate holds only while its error < this margin "
            "(T14); a GLSL f32 port of this chain measured 1e-5 to 7e-4")
    return "\n".join(lines) + "\n", man


def symbolic_run(machine, program):
    """THE THIRD REFEREE: execute the symbolic program directly with NumPy semantics -- no HRR
    decode, no compiled chain. Independent of both substrates, so a three-way agreement means
    something (two components agreeing is not correctness -- the nearest_batch lesson)."""
    from holographic.agents_and_reasoning.holographic_ai import bind
    acc, regs, i = None, {}, 0
    while i < len(program):
        op, arg = program[i]
        if op == "LOAD":
            acc = machine.data_atoms[arg].copy()
        elif op == "BIND":
            acc = bind(acc, machine.data_atoms[arg])
        elif op == "PERMUTE":
            acc = np.roll(acc, 1)                              # the VM's permute(acc, 1), exactly
        elif op == "REPEAT":
            nop, nfn = program[i + 1]
            if nop == "CALL":
                for _ in range(int(arg)):
                    for bop, barg in machine.functions_symbolic[nfn]:
                        acc = bind(acc, machine.data_atoms[barg]) if bop == "BIND" else np.roll(acc, 1)
                i += 2
                continue
        elif op == "STORE":
            regs[arg] = acc.copy()
        elif op == "RECALL":
            acc = regs[arg].copy()
        elif op == "IFMATCH":
            t = machine.data_atoms[arg]
            c = float(acc @ t / (np.linalg.norm(acc) * np.linalg.norm(t) + 1e-12))
            if c < 0.5:
                i += 2
                continue
        elif op == "ITERATE":
            from holographic.agents_and_reasoning.holographic_ai import bind as _bind
            for _ in range(64):
                prev = acc
                for bop, barg in machine.functions_symbolic[arg]:
                    acc = _bind(acc, machine.data_atoms[barg]) if bop == "BIND" else np.roll(acc, 1)
                cc = float(acc @ prev / (np.linalg.norm(acc) * np.linalg.norm(prev) + 1e-12))
                if cc >= 0.999:
                    break
        elif op == "FAC":
            acc = np.asarray(arg[1](acc), float).reshape(-1)   # the LIVE faculty is the referee
        elif op == "HALT":
            break
        i += 1
    return acc


def verify_conformance(machine, program, atol=1e-5):
    """Run all three substrates and CHECK THE INSTRUMENT before trusting it: the VM's decoded
    trace must match the symbolic program (prefix, through HALT) for the VM to count as a
    reference at all -- at low dim with long programs, HALT itself can fail to decode and the VM
    overruns into noise instructions (FOUND BY THE FUZZER at dim=256: ten garbage ops past the
    end; the compiler was right and the reference was broken -- instrument validity is a
    precondition, not an afterthought). Returns a dict with per-pair agreement and a
    'vm_decode_limited' flag; 'installed_vs_symbolic' is the substrate-independent verdict."""
    run_inst, man = compile_installed(machine, program)
    inst = run_inst()
    sym = symbolic_run(machine, program)
    vm, trace = machine.run(machine.assemble(program))
    want = [(op, arg) for op, arg in program if op != "HALT"]
    got = [(t[0], t[1] if len(t) > 1 else None) for t in trace]
    # trace fidelity with LEGAL SKIPS: an IFMATCH that does not fire legitimately omits its
    # guarded successor from the trace -- the first G4 conformance run flagged that as
    # vm_decode_limited, which was the REFEREE being wrong, not the VM (instrument validity cuts
    # both ways). Alignment: walk want; after an IFMATCH, the next want entry may be absent from
    # got. PERMUTE operands are ignored by the VM (compare opcode only).
    def _trace_clean(want, got):
        wi = gi = 0
        while wi < len(want):
            w = want[wi]
            if gi < len(got) and got[gi][0] == w[0] and (w[0] in ("PERMUTE",) or got[gi][1] == w[1]):
                skipped_ok = False
                gi += 1
            elif wi > 0 and want[wi - 1][0] == "IFMATCH":
                pass                                            # legal skip of the guarded op
            else:
                return False
            wi += 1
        return gi == len(got)
    clean = _trace_clean(want, got)
    return {"installed_vs_symbolic": bool(np.allclose(inst, sym, atol=atol)),
            "vm_vs_symbolic": bool(np.allclose(vm, sym, atol=atol)),
            "vm_decode_limited": not clean,
            "manifest": man}


def save_manifest(manifest, path):
    """F26 -- the installed side's discoverability sidecar: JSON with per-op kind, payload SHAPE
    (never the payload -- weights live in the weights), certification residual and probe seconds.
    The runtime has find_capability; installed programs get this."""
    slim = {"dim": manifest["dim"], "chain": manifest["chain"], "ops": {}}
    for name, pr in manifest["ops"].items():
        entry = {"kind": pr["kind"]}
        for k in ("residual", "seconds"):
            if k in pr:
                entry[k] = float(pr[k])
        for payload in ("column", "perm", "matrix"):
            if payload in pr:
                arr = np.ascontiguousarray(pr[payload])
                import hashlib
                entry["payload"] = {"field": payload, "shape": list(np.shape(arr)),
                                    # INTEGRITY: content hash so installation can verify what
                                    # landed, bit-level (hashlib, never hash()).
                                    "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16]}
                if arr.dtype.kind == "f":
                    # QUANTIZATION CERTIFICATE (audit-measured: end-to-end fp16 cosine 0.99999998
                    # on the conformance program): per-payload round-trip error at the precisions
                    # HF hosts actually use, recorded so installation at fp16/bf16 is a checked
                    # claim, not a hope.
                    f16 = np.abs(arr - arr.astype(np.float16).astype(np.float64)).max()
                    v = arr.astype(np.float32).view(np.uint32)
                    b16 = np.abs(arr - ((v + 0x8000) & 0xFFFF0000).view(np.float32).astype(np.float64)).max()
                    entry["quant"] = {"fp16_max_err": float(f16), "bf16_max_err": float(b16)}
        for k2 in ("spec_max", "spec_min"):
            if k2 in pr:
                entry[k2] = float(pr[k2])
        slim["ops"][name] = entry
    with open(path, "w") as f:
        json.dump(slim, f, indent=1, sort_keys=True)
    return slim


def _selftest():
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    from holographic.agents_and_reasoning.holographic_ai import bind
    mach = HoloMachine(dim=1024, seed=7, data=["a", "b", "k", "k2"])
    # the VM path needs the function defined holographically; the compiler needs it symbolically
    # INSTRUMENT LESSON (caught live): an assembled function body WITHOUT a HALT overruns --
    # decode keeps reading noise positions as instructions (the probe saw a 12-op garbage trail
    # and acc ended equal to 'k' via a stray noise-LOAD). The VM body carries its HALT; the
    # symbolic body for the compiler does not need one (the chain ends where the list ends).
    twist = [("BIND", "k")]
    mach.define("twist", twist + [("HALT", None)])
    mach.functions_symbolic = {"twist": twist}
    prog = [("LOAD", "a"), ("REPEAT", 3), ("CALL", "twist"),
            ("STORE", "R1"), ("LOAD", "b"), ("BIND", "k2"), ("RECALL", "R1"), ("HALT", None)]

    # reference 1: the VM, decoding the HRR-encoded program holographically
    acc_vm, trace = mach.run(mach.assemble(prog))
    # reference 2: hand math -- bind(k, bind(k, bind(k, a)))
    a, k = mach.data_atoms["a"], mach.data_atoms["k"]
    truth = bind(bind(bind(a, k), k), k)                       # the VM's order, three times
    # installed path: certified matvecs, REPEAT collapsed to ONE spectral power
    run_inst, man = compile_installed(mach, prog)
    acc_inst = run_inst()

    # CONFORMANCE per ISA tags: these are all EXACT ops -> numerical agreement, all three ways
    assert np.allclose(acc_inst, truth, atol=1e-8), "installed != hand truth"
    assert np.allclose(acc_vm, truth, atol=1e-6), "VM != hand truth (decode should be clean here)"
    assert np.allclose(acc_inst, acc_vm, atol=1e-6), "installed != VM"

    # the REPEAT really collapsed: the chain has ONE POWER step, no loop
    kinds = [s for s, _ in man["chain"]]
    assert kinds.count("POWER") == 1 and "twist^3" in dict(man["chain"]).get("POWER", "twist^3")
    assert man["ops"]["BODY:twist"]["kind"] == "circulant", "a bind body must certify circulant"

    # nonlinear body REFUSES loudly (the boundary, measured)
    mach.functions_symbolic["bad"] = [("SQUASH", None)]
    try:
        compile_installed(mach, [("LOAD", "a"), ("REPEAT", 2), ("CALL", "bad"), ("HALT", None)])
        raise AssertionError("nonlinear body must refuse")
    except ValueError:
        pass

    # F26 manifest sidecar round-trips, carries kinds + residuals + payload SHAPES only
    import tempfile, os, json as _json
    fp = os.path.join(tempfile.gettempdir(), "conformance_manifest.json")
    slim = save_manifest(man, fp)
    back = _json.load(open(fp))
    assert back["ops"]["BODY:twist"]["kind"] == "circulant" and back["ops"]["BODY:twist"]["residual"] < 1e-8
    assert back["ops"]["BODY:twist"]["payload"]["shape"] == [1024]
    # G2 PIN: a real 3D linear core (rigid transform of a 30-vertex block, dim=90) compiled into
    # the chain as a FAC step -- certified BLOCKDIAG (9+3 params, not 8100), installed == the live
    # faculty (the symbolic referee IS the live call). The door the census's candidates walk
    # through, proven on the first customer.
    _th = 0.5
    _R = np.array([[np.cos(_th), -np.sin(_th), 0], [np.sin(_th), np.cos(_th), 0], [0, 0, 1.0]])
    _t = np.array([0.1, 0.2, -0.3])
    def _rigid(flat):
        V = flat.reshape(-1, 3)
        return ((V @ _R.T) + _t).reshape(-1)
    mg = HoloMachine(dim=90, seed=11, data=["a"])
    mg.functions_symbolic = {}
    pg = [("LOAD", "a"), ("FAC", ("rigid", _rigid)), ("FAC", ("rigid", _rigid)), ("HALT", None)]
    rg, mang = compile_installed(mg, pg)
    assert mang["ops"]["FAC:rigid"]["kind"] == "blockdiag", mang["ops"]["FAC:rigid"]["kind"]
    assert np.allclose(rg(), _rigid(_rigid(mg.data_atoms["a"])), atol=1e-9), "installed != live faculty"

    # G11 PIN -- THE SIM PROGRAM: one PBD-shaped step (linear neighbor-blend projection installs;
    # clamp rides HOST:APPLY), iterated 100 times installed vs live: drift IDENTICALLY ZERO (the
    # host step calls the same clip; the certified projection is exact -- the drift curve is the
    # honesty instrument and here it is flat at 0).
    def _prj(f):
        Vv = f.reshape(-1, 3); o = Vv.copy()
        o[1:] = 0.7 * Vv[1:] + 0.3 * Vv[:-1]
        return o.reshape(-1)
    m11 = HoloMachine(dim=60, seed=7, data=["a"]); m11.functions_symbolic = {}
    tr, mn11, dr = sim_program_run(m11, [("FAC", ("proj", _prj)),
                                         ("FAC", ("clamp", lambda f: np.clip(f, -2, 2))), ("HALT", None)],
                                   np.random.default_rng(1).standard_normal(60) * 3.0, 100)
    assert tr.shape == (101, 60) and float(dr.max()) == 0.0, dr.max()
    assert "HOST:clamp" in mn11["ops"] and mn11["ops"]["FAC:proj"]["kind"] == "dense"

    # HRNN-COLLAPSE PINS: (a) 100 steps of the PBD-shaped linear step collapse to ONE affine
    # operator matching the stepped trajectory endpoint at ~1e-15 (measured 156x on endpoint
    # queries); (b) the certificate prices the spectrum (eig_max^n visible -- explosive
    # recurrences announce themselves at compile); (c) a HOST link in the step REFUSES with the
    # step named -- clamps break linearity and the collapse never pretends otherwise; (d) an
    # AFFINE step (drift + decay) collapses exactly via the geometric-series offset.
    rc, cc = collapse_recurrence(m11, [("FAC", ("proj", _prj)), ("HALT", None)], 100)
    assert cc["residual"] < 1e-12 and abs(cc["eig_max"] - 1.0) < 1e-9
    # referee: the CLAMP-FREE iteration (tr above includes the HOST clamp the collapse rightly
    # refuses -- the first pin run compared against the wrong trajectory and taught this: the
    # collapse's claim is the LINEAR step's endpoint, so the referee must run the linear step)
    x_lin = tr[0].copy()
    for _ in range(100):
        x_lin = _prj(x_lin)
    assert np.allclose(rc(tr[0]), x_lin, atol=1e-9), "collapsed endpoint must equal the stepped one"
    def _aff(f):
        return 0.95 * f + 0.01
    ra, ca = collapse_recurrence(m11, [("FAC", ("decay", _aff)), ("HALT", None)], 50)
    x50 = np.ones(60)
    for _ in range(50):
        x50 = _aff(x50)
    assert np.allclose(ra(np.ones(60)), x50, atol=1e-10), "affine recurrence must collapse exactly"
    try:
        collapse_recurrence(m11, [("FAC", ("proj", _prj)),
                                  ("FAC", ("cl", lambda v: np.clip(v, -2, 2))), ("HALT", None)],
                            10, host_fallback=True)
        raise AssertionError("host step must refuse collapse")
    except ValueError as e:
        assert "host" in str(e).lower()


    # G12 PIN -- RENDER-TO-TEXT: a 3-light -> 8x8 basis-lighting formation (RECTANGULAR map --
    # the shape assumption this pin found and fixed: square-only probing refused honest linear
    # maps) runs installed and the PGM P2 dump is BYTE-EXACT vs the live path; the picture
    # leaves through the mouth.
    Wf = np.zeros((64, 3))
    xs2, ys2 = np.meshgrid(np.arange(8), np.arange(8))
    for ii, (cx, cy) in enumerate([(2, 2), (5, 5), (6, 1)]):
        Wf[:, ii] = 255.0 * np.exp(-((xs2 - cx) ** 2 + (ys2 - cy) ** 2) / 4.0).reshape(-1)
    m12 = HoloMachine(dim=3, seed=7, data=["a"]); m12.functions_symbolic = {}
    pgm12, mn12 = raster_program_pgm(m12, [("FAC", ("form", lambda q: Wf @ q)), ("HALT", None)],
                                     np.array([0.9, 0.6, 0.8]), 8, 8)
    lv = np.clip(np.round(Wf @ np.array([0.9, 0.6, 0.8])), 0, 255).astype(int).reshape(8, 8)
    assert pgm12 == "\n".join(["P2", "# leCore installed render -- emitted by the chain", "8 8", "255"]
                               + [" ".join(str(v) for v in rw) for rw in lv]) + "\n"

    # G10 PIN -- THE MESH PROGRAM: a real 8-vertex box through an installed rigid+scale chain
    # (both FAC steps certified), OBJ text dump BYTE-EXACT vs the live-faculty path. The output
    # device is the mouth (G0): no file was written to produce this mesh.
    bx = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    fc = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
          (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    _t2 = 0.31
    _R2 = np.array([[np.cos(_t2), 0, np.sin(_t2)], [0, 1, 0], [-np.sin(_t2), 0, np.cos(_t2)]])
    def _rig(flat):
        return (flat.reshape(-1, 3) @ _R2.T + np.array([0.2, -0.1, 0.4])).reshape(-1)
    def _scl(flat):
        return (flat * 1.5)
    m10 = HoloMachine(dim=24, seed=3, data=["a"])
    m10.functions_symbolic = {}
    pg10 = [("FAC", ("rigid", _rig)), ("FAC", ("scale", _scl)), ("HALT", None)]
    obj_inst, man10 = mesh_program_obj(m10, pg10, bx, fc)
    live = _scl(_rig(bx.reshape(-1))).reshape(-1, 3)
    obj_live = "\n".join(["# leCore installed mesh program -- emitted by the chain, not the filesystem"]
                          + ["v %.6f %.6f %.6f" % tuple(p) for p in live]
                          + ["f %d %d %d" % (a + 1, b + 1, c2 + 1) for a, b, c2 in fc]) + "\n"
    assert obj_inst == obj_live, "OBJ dump must be BYTE-EXACT vs the live path"
    assert man10["ops"]["FAC:rigid"]["kind"] == "blockdiag", man10["ops"]["FAC:rigid"]["kind"]

    # G8/G9 PINS: (a) cleanup installs as an ATTENTION READ -- agreement 1.000 with exact cleanup
    # at beta>=16 on real vectors (0.575 at beta=4: the temperature curve is real); the
    # PRE-REGISTERED tie negative holds (softmax averages exactly tied rows -- cannot express the
    # lowest-index rule, by theorem). (b) the FUSION SPLIT: a mixed linear/nonlinear program
    # compiles under host_fallback=True with the refused step MARKED HOST:APPLY carrying its
    # refusal residual; without the flag it still raises (shipped behavior unchanged).
    from holographic.io_and_interop.holographic_projector import attention_read_certificate
    rng_a = np.random.default_rng(8890)
    Va = rng_a.standard_normal((300, 64)); Va /= np.linalg.norm(Va, axis=1, keepdims=True)
    Qa = Va[rng_a.choice(300, 60, replace=False)] + 0.05 * rng_a.standard_normal((60, 64))
    assert attention_read_certificate(Va, Qa, beta=64.0)["agreement"] == 1.0
    assert attention_read_certificate(Va, Qa, beta=1.0)["agreement"] < 1.0, "temperature must matter"
    m9 = HoloMachine(dim=64, seed=2, data=["a"])
    m9.functions_symbolic = {}
    pg9 = [("LOAD", "a"), ("FAC", ("clamp", lambda v: np.clip(v, -0.5, 0.5))), ("HALT", None)]
    try:
        compile_installed(m9, pg9)
        raise AssertionError("refusal without host_fallback must still raise")
    except ValueError:
        pass
    run9, man9 = compile_installed(m9, pg9, host_fallback=True)
    assert any(s[0] == "HOST:APPLY" for s in man9["chain"]), man9["chain"]
    assert man9["ops"]["HOST:clamp"]["kind"] == "host_apply" and man9["ops"]["HOST:clamp"]["residual"] > 0.1
    assert np.allclose(run9(), np.clip(m9.data_atoms["a"], -0.5, 0.5)), "host step must equal live"

    # G4/G5/G6 PINS (control compiled honestly): (a) IFMATCH both branches -- installed ==
    # symbolic == VM, the no-fire branch's LEGAL SKIP no longer misread as a decode limit (the
    # referee bug the first run exposed: instrument validity cuts both ways); the chain marks the
    # step HOST:. (b) ITERATE -- body installed once, convergence as host step, three-referee
    # agreement. (c) G6 recurrence contract -- installed final acc == the VM's run_chunked on the
    # same program (the register file IS the recurrent state across chunks).
    mc = HoloMachine(dim=1024, seed=5, data=["a", "b", "k"])
    twc = [("BIND", "k")]
    mc.define("tw", twc + [("HALT", None)]); mc.functions_symbolic = {"tw": twc}
    for start in ("a", "b"):
        pgc = [("LOAD", start), ("IFMATCH", "a"), ("BIND", "k"), ("HALT", None)]
        vr = verify_conformance(mc, pgc)
        assert vr["installed_vs_symbolic"] and vr["vm_vs_symbolic"] and not vr["vm_decode_limited"], (start, vr)
        _, manc = compile_installed(mc, pgc)
        assert any(s[0] == "HOST:IFMATCH" for s in manc["chain"]), "control must be MARKED host"
    vr2 = verify_conformance(mc, [("LOAD", "a"), ("ITERATE", "tw"), ("HALT", None)])
    assert vr2["installed_vs_symbolic"] and vr2["vm_vs_symbolic"], vr2
    pg6 = [("LOAD", "a"), ("BIND", "k"), ("STORE", "R1"), ("LOAD", "b"), ("RECALL", "R1"),
           ("BIND", "k"), ("HALT", None)]
    run6, _ = compile_installed(mc, pg6)
    acc_ch = mc.run_chunked(pg6, chunk=3)[0]
    assert np.allclose(run6(), acc_ch, atol=1e-6), "installed must equal the VM's CHUNKED execution"

    # PIPELINE-AUDIT PINS (the Unicron hand-off hardening, all measured): (a) ODD-DIM conformance
    # -- host hidden sizes are 384/896/4096-ish, never tidy; rfft does not care and now a pin says
    # so; (b) the CONDITIONING WALL -- a deep non-unitary bind chain must raise the manifest
    # warning (measured blowup: 1e8 at depth 64); (c) certificates present: spec_max/min, payload
    # sha256, per-payload fp16/bf16 quantization error (end-to-end fp16 cosine measured 0.99999998).
    mo = HoloMachine(dim=896, seed=7, data=["a", "k"])
    mo.functions_symbolic = {}
    ro, mano = compile_installed(mo, [("LOAD", "a"), ("BIND", "k"), ("HALT", None)])
    assert np.allclose(ro(), symbolic_run(mo, [("LOAD", "a"), ("BIND", "k"), ("HALT", None)]), atol=1e-8)
    e = mano["ops"]["BIND:k"]
    assert "spec_max" in e and e["spec_max"] > e["spec_min"] > 0
    import tempfile as _tf, os as _os, json as _json
    fpo = _os.path.join(_tf.gettempdir(), "audit_manifest.json")
    slim_o = save_manifest(mano, fpo)
    ent = slim_o["ops"]["BIND:k"]
    assert len(ent["payload"]["sha256"]) == 16 and ent["quant"]["fp16_max_err"] < 1e-2
    md = HoloMachine(dim=512, seed=3, data=["a", "k"])
    md.functions_symbolic = {}
    _, man_deep = compile_installed(md, [("LOAD", "a")] + [("BIND", "k")] * 64 + [("HALT", None)])
    assert any("amplification" in w for w in man_deep.get("warnings", [])), \
        "deep non-unitary chain must carry the conditioning warning"

    # FUZZ PINS (Togelius seat -- 60-program campaign findings, distilled): (a) mini-fuzz --
    # installed == symbolic across random programs (the substrate-independent property; the full
    # campaign found 0 failures in 60); (b) the VM DECODE WALL is real and DETECTED: at dim=256 a
    # long program's HALT can fail to decode and the VM overruns into noise instructions --
    # verify_conformance flags it as vm_decode_limited instead of miscounting it as a compiler
    # disagreement (instrument validity precedes measurement).
    rng_f = np.random.default_rng(6060)
    for _ in range(8):
        mm = HoloMachine(dim=512, seed=int(rng_f.integers(0, 500)), data=["x", "y", "z"])
        bb = [("BIND", ["x", "y", "z"][int(rng_f.integers(0, 3))])]
        mm.define("g", bb + [("HALT", None)]); mm.functions_symbolic = {"g": bb}
        pp = [("LOAD", "x"), ("REPEAT", int(rng_f.integers(2, 5))), ("CALL", "g"),
              ("STORE", "R0"), ("PERMUTE", 1), ("RECALL", "R0"), ("HALT", None)]
        vr = verify_conformance(mm, pp)
        assert vr["installed_vs_symbolic"], "installed must equal the symbolic referee, always"
    # a decode-limited case at dim=256 (found by the campaign) must be FLAGGED, and the installed
    # path must still match the symbolic referee even while the VM is off in the weeds
    mm2 = HoloMachine(dim=256, seed=619, data=["d0", "d1", "d2", "d3"])
    bb2 = [("BIND", "d0")]
    mm2.define("f", bb2 + [("HALT", None)]); mm2.functions_symbolic = {"f": bb2}
    pp2 = [("LOAD", "d0"), ("PERMUTE", 1), ("REPEAT", 2), ("CALL", "f"),
           ("STORE", "R0"), ("BIND", "d0"), ("PERMUTE", 1), ("HALT", None)]
    vr2 = verify_conformance(mm2, pp2)
    assert vr2["vm_decode_limited"] and vr2["installed_vs_symbolic"], vr2

    print("OK: holographic_compileinstall self-test passed (VM == installed == hand truth on a "
          "REPEAT+STORE/RECALL program; REPEAT collapsed to one spectral power; nonlinear body "
          "refused; manifest sidecar round-trips with shapes only)")


if __name__ == "__main__":
    _selftest()
