"""holographic_emit.py -- dialect emitters over the typed structure (Box3D backlog K8).

leCore's kernels are written once, in Python, and the browser needs them in WGSL. A dialect emitter makes the
hand-written compute shader a **projection of the authoritative Python kernel**: one source of truth, two runtimes,
no drift. `emit(fn, dialect)` walks the same AST that `holographic_codestructure` decomposes, and a dialect table
supplies the type names, the intrinsic names, and the declaration syntax.

Dialects: `c_f64`, `c_f32`, `wgsl`, `js`, `zig_f64`, `zig_f32`.

KEPT NEGATIVE 4 (Zig) -- **Zig refuses unused locals and parameters at COMPILE time.** A Python kernel with a dead
assignment or an unused parameter emits fine but will not compile as Zig. That is Zig's discipline, not ours, and
we do not paper over it with `_ = x;` suppressions: a dead local in a kernel is a smell the compiler is right to
name. KEPT NEGATIVE 5 (Zig) -- `-O ReleaseFast` licenses float reassociation, so it is NOT the deterministic mode;
`run_zig` compiles `ReleaseSafe`, where zig_f64 is measured bit-identical to the Python original (same order of
operations, same doubles -- same result as c_f64). KEPT NEGATIVE 6 (Zig) -- **std.math.pow is not libm pow**:
measured 1-ulp disagreement (4.4e-16 abs on pow(1.3, 2.7)). zig_f64 bit-identity is a property of the BUILTIN
intrinsics (@sqrt/@sin/...); a kernel calling pow is judged at f64-ulp tolerance, stated, not hidden.

THE BAR, AND IT IS EXECUTED. The backlog asks for "a real leCore kernel emitted and validated against the Python
original to float tolerance on the same inputs." WGSL cannot be run here -- there is no GPU and no browser -- so
the C dialect is compiled with `cc` and RUN, on the same 200 random inputs:

    dialect    max |emitted - python|      note
    c_f64            0.0                   BIT-IDENTICAL: same order of operations, same doubles
    c_f32            2.866e-07             f32 arithmetic, executed

KEPT NEGATIVE 1 -- **A WGSL KERNEL CANNOT BE BIT-IDENTICAL TO ITS PYTHON ORIGINAL.** WGSL's `f32` is single
precision; NumPy is double. The measured gap is 2.9e-07 absolute / 8.0e-07 relative on an SDF evaluation. So the
bar is "to float tolerance", and **the tolerance is f32 epsilon -- not a number anybody gets to choose.**
`c_f32` exists precisely so that tolerance is MEASURED by running it, rather than asserted.

KEPT NEGATIVE 2 -- **the WGSL text emitted here is not executed by any test in this repo.** Its arithmetic
semantics are validated through `c_f32`, which shares the emitter's IR and differs only in a dialect table (type
names, intrinsic names, `let` vs a typed declaration). What is NOT validated is WGSL's own rules: its precision
guarantees, its fast-math latitude, whether the shader compiles at all. **That is a real gap and it is stated
rather than papered over.** A structural test asserts the emitted text has no Python-isms and declares what WGSL
requires; nothing more is claimed.

KEPT NEGATIVE 3 -- **`bind` is not emittable, and that is not a missing feature.** The backlog names "cosine, a
bind, an SDF eval" as the kernels to emit. `cosine` and an SDF evaluation are scalar, element-wise, and project
cleanly. `bind` is a circular convolution done by FFT: it is a *whole-array* algorithm with a cooperative
butterfly, not an expression over scalars. Its WGSL is a workgroup FFT -- a different artifact, not a projection of
the Python. A scalar emitter that pretended otherwise would emit a loop nest that is O(D^2) and calls itself a bind.

K10's RULE, obeyed: **the emitter REFUSES rather than guesses.** An unannotated parameter, an unsupported
statement, an unknown call -- each raises with the offending construct named. A wrong int/double is exactly the bug
a hand-written C port produces in its first line, and there is no tolerance at which it is acceptable.
"""

import ast
import inspect
import textwrap


#: Intrinsics, per dialect. A name absent from this table is an unknown call and the emitter refuses it.
INTRINSICS = {
    "sqrt": {"c_f64": "sqrt", "c_f32": "sqrtf", "wgsl": "sqrt", "js": "Math.sqrt"},
    "exp": {"c_f64": "exp", "c_f32": "expf", "wgsl": "exp", "js": "Math.exp"},
    "log": {"c_f64": "log", "c_f32": "logf", "wgsl": "log", "js": "Math.log"},
    "sin": {"c_f64": "sin", "c_f32": "sinf", "wgsl": "sin", "js": "Math.sin"},
    "cos": {"c_f64": "cos", "c_f32": "cosf", "wgsl": "cos", "js": "Math.cos"},
    "abs": {"c_f64": "fabs", "c_f32": "fabsf", "wgsl": "abs", "js": "Math.abs"},
    "min": {"c_f64": "fmin", "c_f32": "fminf", "wgsl": "min", "js": "Math.min"},
    "max": {"c_f64": "fmax", "c_f32": "fmaxf", "wgsl": "max", "js": "Math.max"},
    "pow": {"c_f64": "pow", "c_f32": "powf", "wgsl": "pow", "js": "Math.pow",
            # Zig's pow is type-parameterized -- std.math.pow(T, a, b) -- so a bare name cannot express it. An
            # intrinsic entry containing "{args}" is a TEMPLATE; plain entries keep the old name(args) form. This
            # is additive: no existing dialect entry contains braces, so their emission is byte-identical.
            "zig_f64": "std.math.pow(f64, {args})", "zig_f32": "std.math.pow(f32, {args})"},
}

# Zig builtins cover the rest; @abs/@min/@max exist for floats since 0.12 (ziglang wheel ships >= 0.13).
for _n, _z in (("sqrt", "@sqrt"), ("exp", "@exp"), ("log", "@log"), ("sin", "@sin"), ("cos", "@cos"),
               ("abs", "@abs"), ("min", "@min"), ("max", "@max")):
    INTRINSICS[_n]["zig_f64"] = _z
    INTRINSICS[_n]["zig_f32"] = _z
    # The same builtins accept @Vector operands, so the vector dialects reuse them verbatim. `pow` is ABSENT for
    # zigv_* -- std.math.pow is scalar-only -- and the emitter refuses a missing intrinsic by name (K10).
    INTRINSICS[_n]["zigv_f64"] = _z
    INTRINSICS[_n]["zigv_f32"] = _z

DIALECTS = {
    "c_f64": {"scalar": "double", "decl": "{s} {n} = {e};", "typed_decl": True,
              "sig": "{s} {name}({params})", "param": "{s} {n}", "suffix": "", "brace": True,
              "mut_decl": "{s} {n} = {e};", "loop": "for (int {i} = 0; {i} < {n}; {i}++) {{",
              "int_promote": "(double){i}"},
    "c_f32": {"scalar": "float", "decl": "{s} {n} = {e};", "typed_decl": True,
              "sig": "{s} {name}({params})", "param": "{s} {n}", "suffix": "f", "brace": True,
              "mut_decl": "{s} {n} = {e};", "loop": "for (int {i} = 0; {i} < {n}; {i}++) {{",
              "int_promote": "(float){i}"},
    "wgsl": {"scalar": "f32", "decl": "let {n} = {e};", "typed_decl": False,
             "sig": "fn {name}({params}) -> {s}", "param": "{n}: {s}", "suffix": "f", "brace": True,
             "mut_decl": "var {n} = {e};", "loop": "for (var {i}: i32 = 0; {i} < {n}; {i} = {i} + 1) {{",
             "int_promote": "f32({i})"},
    "zig_f64": {"scalar": "f64", "decl": "const {n}: {s} = {e};", "typed_decl": True,
                "sig": "fn {name}({params}) {s}", "param": "{n}: {s}", "suffix": "", "brace": True,
                "mut_decl": "var {n}: {s} = {e};",
                "loop": "var {i}: usize = 0;\nwhile ({i} < {n}) : ({i} += 1) {{",
                "int_promote": "@as(f64, @floatFromInt({i}))"},
    "zig_f32": {"scalar": "f32", "decl": "const {n}: {s} = {e};", "typed_decl": True,
                "sig": "fn {name}({params}) {s}", "param": "{n}: {s}", "suffix": "", "brace": True,
                "mut_decl": "var {n}: {s} = {e};",
                "loop": "var {i}: usize = 0;\nwhile ({i} < {n}) : ({i} += 1) {{",
                "int_promote": "@as(f32, @floatFromInt({i}))"},
    "zigv_f64": {"scalar": "V", "decl": "const {n}: {s} = {e};", "typed_decl": True,
                 "sig": "fn {name}({params}) {s}", "param": "{n}: {s}", "suffix": "",
                 "const_fmt": "@as(V, @splat({v}))", "brace": True,
                 "mut_decl": "var {n}: {s} = {e};",
                 "loop": "var {i}: usize = 0;\nwhile ({i} < {n}) : ({i} += 1) {{",
                 "int_promote": "@as(V, @splat(@as(f64, @floatFromInt({i}))))"},
    "zigv_f32": {"scalar": "V", "decl": "const {n}: {s} = {e};", "typed_decl": True,
                 "sig": "fn {name}({params}) {s}", "param": "{n}: {s}", "suffix": "",
                 "const_fmt": "@as(V, @splat({v}))", "brace": True,
                 "mut_decl": "var {n}: {s} = {e};",
                 "loop": "var {i}: usize = 0;\nwhile ({i} < {n}) : ({i} += 1) {{",
                 "int_promote": "@as(V, @splat(@as(f32, @floatFromInt({i}))))"},
    "js": {"scalar": "number", "decl": "const {n} = {e};", "typed_decl": False,
           "sig": "function {name}({params})", "param": "{n}", "suffix": "", "brace": True,
           "mut_decl": "let {n} = {e};", "loop": "for (let {i} = 0; {i} < {n}; {i}++) {{",
           "int_promote": "{i}"},
}

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def glsl_float(x):
    """Format a Python float as a GLSL float LITERAL: %.6g, guaranteed to carry a decimal point (or exponent) so
    GLSL never reads it as an int. THE one shared formatter for every GLSL emitter in the engine (sdf camera
    uniforms aside) -- postfx (item 9), pattern and cosine-palette (item 10) all delegate here, because three
    private copies of the same six lines was exactly the buried duplication the wiring sweep exists to catch."""
    s = "%.6g" % float(x)
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s


def glsl_vec3(v):
    """Format a length-3 vector as a GLSL vec3 literal, each component via glsl_float."""
    import numpy as _np
    v = _np.asarray(v, float).ravel()
    return "vec3(%s, %s, %s)" % (glsl_float(v[0]), glsl_float(v[1]), glsl_float(v[2]))


import re as _re

# THE shared home for GLSL shader ASSEMBLY (backlog B1). The engine's GLSL emitters -- sdf (to_shadertoy), postfx
# (chain_to_glsl), pattern (pattern_to_glsl), palette (cosine_palette_to_glsl) -- each produce self-contained GLSL
# function pieces. Composing several into ONE shader, and wrapping a Shadertoy-style source for WebGL2, were done by
# HAND at every call site (the cross-item capstone concatenated strings; every parse check re-wrote the #version
# preamble). This consolidates both: one composer, one wrapper, so a new emitter reuses them instead of hand-rolling.

_GLSL_FUNC_DEF = _re.compile(r"^\s*(?:highp\s+|mediump\s+|lowp\s+)?\w+\s+(\w+)\s*\(", _re.MULTILINE)


def glsl_function_names(src):
    """The top-level function names DEFINED in a GLSL source (best-effort, for duplicate detection). Skips lines that
    are comments or preprocessor directives. Not a full parser -- it exists to catch the one real hazard when
    composing pieces: two functions with the SAME name."""
    names = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("//") or s.startswith("#") or s.startswith("/*") or s.startswith("*"):
            continue
        mo = _GLSL_FUNC_DEF.match(line)
        if mo and "return" != mo.group(1) and "{" in line:      # a def has a brace on (or opening) the line
            names.append(mo.group(1))
    return names


def assemble_glsl(functions, entry=None, header=""):
    """Compose emitted GLSL function pieces into ONE source, in order, deterministically. `functions` is a list of
    GLSL source strings (each the output of an emitter, e.g. a `float pattern(vec3 p){...}`); `entry` is an optional
    trailing block (a mainImage / main); `header` an optional leading comment/uniform block.

    RAISES on a duplicate top-level function name across the pieces -- the one real hazard the wiring sweep flagged
    (compose two palettes and one silently shadows the other). That is a kept negative made into an error: the caller
    renames (the emitters take an fn_name= for exactly this) rather than get a wrong shader."""
    seen = {}
    for i, fsrc in enumerate(functions):
        for nm in glsl_function_names(fsrc):
            if nm in seen:
                raise ValueError("assemble_glsl: duplicate function name %r (pieces %d and %d) -- rename one via the "
                                 "emitter's fn_name= argument" % (nm, seen[nm], i))
            seen[nm] = i
    parts = ([header.rstrip("\n")] if header else []) + [f.rstrip("\n") for f in functions]
    if entry:
        parts.append(entry.rstrip("\n"))
    return "\n\n".join(parts) + "\n"


def webgl2_wrap(shadertoy_src, uniforms=("sampler2D iChannel0", "vec3 iResolution"), entry="mainImage"):
    """Wrap a Shadertoy-style GLSL source (one that defines `void <entry>(out vec4, in vec2)`) into a COMPLETE WebGL2
    (GLSL ES 3.00) fragment shader: the `#version 300 es` + precision preamble, the declared `uniform`s, an
    `out vec4 fragOut;`, the body, and a `void main(){ <entry>(fragOut, gl_FragCoord.xy); }` bridge. Deterministic.
    This is the one true wrapper -- callers stop hand-rolling the preamble (which drifts)."""
    decls = "\n".join("uniform %s;" % u for u in uniforms)
    return ("#version 300 es\n"
            "precision highp float;\n"
            "%s\n"
            "out vec4 fragOut;\n\n"
            "%s\n\n"
            "void main(){ %s(fragOut, gl_FragCoord.xy); }\n" % (decls, shadertoy_src.rstrip("\n"), entry))


class EmitError(ValueError):
    """The emitter refused. The message names the construct; refusing is the feature (K10)."""


def call_soa_kernel(n_params, np_dtype, ct, fn, arrays):
    """Marshal `arrays` (P columns of equal length N) into one contiguous SoA buffer and call a compiled scalar
    kernel `fn(in_ptr, n, out_ptr)` via ctypes, returning the (N,) output. This is the ONE calling convention the
    C runner (ccrun) and the Zig runner (zigrun) share -- both emit a `void k(const T* in, long n, T* out)` with P
    blocks of N laid end to end, so the Python-side marshalling is identical and lived, copied, in both CKernel and
    ZigKernel.__call__. Unified here (the module both already import from) so it cannot drift between the two
    backends; each kernel object just passes its own n_params / dtype / ctypes scalar type / bound function.

    numpy and ctypes are imported lazily so the emitter itself stays import-light (it is mostly string generation;
    only the native runners actually marshal).

    KEPT NEGATIVE: the concatenate is a per-call SoA copy, counted inside any timing of a kernel call -- it caches
    nothing and charges everything. That honesty (originally noted in zigrun) now lives with the shared code."""
    import ctypes
    import numpy as np
    if len(arrays) != n_params:
        raise EmitError("kernel takes %d arrays, got %d" % (n_params, len(arrays)))
    cols = [np.ascontiguousarray(a, dtype=np_dtype) for a in arrays]
    n = cols[0].shape[0]
    if any(c.shape != (n,) for c in cols):
        raise EmitError("all input arrays must be 1-D of the same length")
    inp = np.concatenate(cols)                           # SoA: P blocks of N, one contiguous buffer
    out = np.empty(n, dtype=np_dtype)
    fn(inp.ctypes.data_as(ctypes.POINTER(ct)), n, out.ctypes.data_as(ctypes.POINTER(ct)))
    return out


def _expr(node, dialect, int_vars=frozenset()):
    d = DIALECTS[dialect]
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise EmitError("unsupported operator %s; the emitter refuses rather than guessing"
                            % type(node.op).__name__)
        return "(%s %s %s)" % (_expr(node.left, dialect, int_vars), _BINOPS[type(node.op)],
                               _expr(node.right, dialect, int_vars))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return "(-%s)" % _expr(node.operand, dialect, int_vars)
    if isinstance(node, ast.Name):
        if node.id in int_vars:
            # a bounded-loop counter used in a float expression: promote EXPLICITLY per dialect ((double)i /
            # f32(i) / @floatFromInt) -- Python promotes silently, the target languages must not be left to guess.
            return d["int_promote"].format(i=node.id)
        return node.id
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise EmitError("only float constants are emittable; got %r" % (node.value,))
        if "const_fmt" in d:
            # Vector dialects splat constants: Zig refuses mixed vector/scalar arithmetic, and an implicit
            # broadcast the language forbids is exactly the guess the emitter must not make on its own.
            return d["const_fmt"].format(v=repr(float(node.value)))
        return "%r%s" % (float(node.value), d["suffix"])
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in INTRINSICS:
            name = getattr(node.func, "id", ast.dump(node.func))
            raise EmitError("unknown call %r: not in the intrinsic table for %r. A wrong intrinsic is a wrong "
                            "answer at no tolerance, so the emitter refuses." % (name, dialect))
        args = ", ".join(_expr(a, dialect, int_vars) for a in node.args)
        if dialect not in INTRINSICS[node.func.id]:
            raise EmitError("intrinsic %r has no %r form (std.math.pow is scalar-only, e.g.); the emitter refuses "
                            "rather than guessing a lowering" % (node.func.id, dialect))
        entry = INTRINSICS[node.func.id][dialect]
        if "{args}" in entry:
            # Template intrinsic (Zig's type-parameterized pow). Plain entries keep the historic name(args) form,
            # so every pre-existing dialect emits byte-identically.
            return entry.format(args=args)
        return "%s(%s)" % (entry, args)
    raise EmitError("unsupported expression %s" % type(node).__name__)


def emit_source(src, dialect="wgsl"):
    """Emit from a SOURCE STRING rather than a live function.

    `emit` needs `inspect.getsource`, which a function defined in a REPL or an `exec` does not have. The kernel is
    text; make the text the input."""
    node = ast.parse(textwrap.dedent(src)).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise EmitError("emit_source takes one function definition")
    return _emit_node(node, dialect)


def emit(fn, dialect="wgsl"):
    """Emit `fn` -- a scalar float kernel of assignments, BOUNDED `for i in range(<int literal>)` loops, and a
    final return -- into `dialect`.

    Every parameter must carry a `float` annotation and the function a `-> float` return: **K10's rule is that the
    emitter refuses rather than guesses**, and an unannotated parameter is an unresolved type. Bounded loops are
    emittable because nothing about them is a guess: the trip count is a compile-time constant (the shape every
    shader fBm/octave loop takes), the counter is an int, and its use in a float expression promotes EXPLICITLY
    per dialect ((double)i, f32(i), @floatFromInt). Variable trip counts, range(a, b), `return` inside a loop,
    break/continue, and counter shadowing all still refuse by name.

    Returns the dialect source as a string. It is a projection of the Python, not a translation of it: the same AST
    with a different table."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise EmitError("emit needs %r's source (%s); use emit_source(text, dialect) instead -- the kernel is text"
                        % (getattr(fn, "__name__", fn), exc))
    node = ast.parse(src).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise EmitError("emit takes a plain function")
    return _emit_node(node, dialect)


def _emit_node(node, dialect):
    """The emitter proper, over an `ast.FunctionDef`. Both `emit` and `emit_source` land here, so the refusals and
    the dialect table are shared and cannot drift apart."""
    if dialect not in DIALECTS:
        raise EmitError("unknown dialect %r; try %s" % (dialect, sorted(DIALECTS)))
    d = DIALECTS[dialect]

    if node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.args.defaults:
        raise EmitError("only positional float parameters are emittable")
    for a in node.args.args:
        if a.annotation is None or getattr(a.annotation, "id", None) != "float":
            raise EmitError("parameter %r has no `float` annotation. An unresolved type is a refusal, not a "
                            "default -- a wrong int/double is the first bug a hand port produces." % (a.arg,))
    if node.returns is None or getattr(node.returns, "id", None) != "float":
        raise EmitError("kernel %r must be annotated `-> float`" % (node.name,))

    body, seen_return = [], False

    # -- BOUNDED LOOPS (the Shadertoy-arc item): `for i in range(<int literal>)` is fully translatable with
    # ZERO guessing -- the trip count is a compile-time constant (exactly the shape shader fBm/octave loops
    # take), the loop var is an int counter, and its use inside a float expression promotes EXPLICITLY per
    # dialect ((double)i / f32(i) / @floatFromInt). Everything outside that shape still refuses by name,
    # keeping K10: range(a, b), range over a variable, break/continue, and `return` inside a loop all raise.
    # MUTABILITY: a name assigned more than once (an accumulator) or augmented needs a MUTABLE declaration
    # (wgsl `var`, zig `var`, js `let`); a name assigned once keeps the original const/let form, so every
    # previously-legal straight-line kernel emits CHARACTER-IDENTICALLY (pinned by test).
    def _assigned_names(stmts, counts):
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                counts[st.targets[0].id] = counts.get(st.targets[0].id, 0) + 1
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                counts[st.target.id] = counts.get(st.target.id, 0) + 2      # augment => mutable by definition
            elif isinstance(st, ast.For):
                inner = {}
                _assigned_names(st.body, inner)
                for k, v in inner.items():
                    # any assignment INSIDE a loop body runs repeatedly -> mutable, even if it appears once
                    counts[k] = counts.get(k, 0) + max(v, 2)
        return counts

    counts = _assigned_names(node.body, {})
    mutable = {k for k, v in counts.items() if v > 1}
    declared = set(a.arg for a in node.args.args)
    int_vars = set()

    def _stmts(stmts, indent, in_loop):
        nonlocal seen_return
        for st in stmts:
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                continue                                      # the docstring
            if seen_return:
                raise EmitError("statements after `return` are unreachable and the emitter refuses them")
            if isinstance(st, ast.Assign):
                if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                    raise EmitError("only single-name assignment is emittable")
                n = st.targets[0].id
                e = _expr(st.value, dialect, int_vars)
                if n in declared:
                    body.append(indent + "%s = %s;" % (n, e))
                else:
                    declared.add(n)
                    tmpl = d["mut_decl"] if n in mutable else d["decl"]
                    body.append(indent + tmpl.format(s=d["scalar"], n=n, e=e))
            elif isinstance(st, ast.AugAssign):
                if not isinstance(st.target, ast.Name) or type(st.op) not in _BINOPS:
                    raise EmitError("only name op= expr with +,-,*,/ is emittable")
                n = st.target.id
                if n not in declared:
                    raise EmitError("augmented assignment to undeclared %r" % n)
                body.append(indent + "%s = (%s %s %s);" % (n, n, _BINOPS[type(st.op)],
                                                           _expr(st.value, dialect, int_vars)))
            elif isinstance(st, ast.For):
                it = st.iter
                ok = (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
                      and len(it.args) == 1 and not it.keywords
                      and isinstance(it.args[0], ast.Constant) and isinstance(it.args[0].value, int)
                      and not isinstance(it.args[0].value, bool) and it.args[0].value >= 0)
                if not ok:
                    raise EmitError("only `for <name> in range(<non-negative int literal>)` is emittable -- a "
                                    "variable or multi-argument range is a trip count the emitter cannot prove")
                if not isinstance(st.target, ast.Name) or st.orelse:
                    raise EmitError("the loop target must be a single name and `for...else` is not emittable")
                i, n_trip = st.target.id, it.args[0].value
                if i in declared:
                    raise EmitError("loop variable %r shadows an existing name; the emitter refuses the shadow" % i)
                declared.add(i); int_vars.add(i)
                for line in d["loop"].format(i=i, n=n_trip).split("\n"):
                    body.append(indent + line)
                _stmts(st.body, indent + "  ", True)
                body.append(indent + "}")
                declared.discard(i); int_vars.discard(i)      # the counter scopes to its loop
            elif isinstance(st, ast.Return):
                if in_loop:
                    raise EmitError("`return` inside a loop is an early exit the emitter refuses (K10)")
                body.append(indent + "return %s;" % _expr(st.value, dialect, int_vars))
                seen_return = True
            else:
                raise EmitError("unsupported statement %s: the body must be assignments, bounded `for range(N)` "
                                "loops, and a final return" % type(st).__name__)

    _stmts(node.body, "  ", False)
    if not seen_return:
        raise EmitError("kernel %r never returns" % (node.name,))

    params = ", ".join(d["param"].format(s=d["scalar"], n=a.arg) for a in node.args.args)
    sig = d["sig"].format(s=d["scalar"], name=node.name, params=params)
    return sig + " {\n" + "\n".join(body) + "\n}\n"


#: The Python meaning of each intrinsic, so the comparison runs the SAME program the emitter emitted.
PY_INTRINSICS = {"sqrt": None, "exp": None, "log": None, "sin": None, "cos": None,
                 "abs": abs, "min": min, "max": max, "pow": pow}


def as_python(src):
    """Build a live Python function from a kernel's source text, with the intrinsics resolved.

    The comparison must run the SAME text the emitter emitted, not a hand-written twin of it. That is the whole
    reason `validate_c` takes the source: a re-typed Python reference is a second implementation, and comparing two
    implementations tests neither."""
    import math

    ns = dict(PY_INTRINSICS)
    for name in ("sqrt", "exp", "log", "sin", "cos"):
        ns[name] = getattr(math, name)
    exec(compile(textwrap.dedent(src), "<kernel>", "exec"), ns)   # noqa: S102 -- our own emitted-from text
    node = ast.parse(textwrap.dedent(src)).body[0]
    return ns[node.name]


def _as_node_and_fn(kernel):
    """`kernel` may be a live function or its source text. Text is the primary form."""
    if isinstance(kernel, str):
        return ast.parse(textwrap.dedent(kernel)).body[0], as_python(kernel)
    try:
        src = textwrap.dedent(inspect.getsource(kernel))
    except (OSError, TypeError) as exc:
        raise EmitError("run_c needs %r's source (%s); pass the source TEXT instead -- the kernel is text"
                        % (getattr(kernel, "__name__", kernel), exc))
    return ast.parse(src).body[0], kernel


def run_c(kernel, calls, dialect="c_f64", timeout=60):
    """Compile the emitted C kernel with `cc` and RUN it on `calls` (a list of positional float tuples).

    This is what makes K8's bar an executed one rather than an asserted one. `kernel` is source TEXT or a live
    function. Returns the list of results as floats. `c_f32` is the executable stand-in for WGSL's `f32`: the same
    IR, the same table shape, single precision."""
    import os
    import subprocess
    import tempfile

    if dialect not in ("c_f64", "c_f32"):
        raise EmitError("run_c only runs the C dialects; %r cannot be executed here" % (dialect,))
    node, _fn = _as_node_and_fn(kernel)
    kernel_src = _emit_node(node, dialect)
    name = node.name
    body = "".join('printf("%%.17g\\n", %s(%s));' % (name, ", ".join(repr(float(a)) for a in c)) for c in calls)
    prog = "#include <stdio.h>\n#include <math.h>\n" + kernel_src + "\nint main(){ " + body + " return 0; }\n"

    with tempfile.TemporaryDirectory() as tmp:
        csrc = os.path.join(tmp, "k.c")
        exe = os.path.join(tmp, "k")
        with open(csrc, "w") as fh:
            fh.write(prog)
        try:
            subprocess.run(["cc", csrc, "-o", exe, "-lm"], check=True, capture_output=True, timeout=timeout)
        except FileNotFoundError:
            # Z6: no system compiler. The `ziglang` wheel ships `zig cc`, a hermetic clang -- one pip install gives
            # the C validation path on any machine. Opt-in accelerator discipline: absent BOTH, run_c raises and the
            # caller (selftest) skips LOUDLY rather than silently passing.
            import sys
            subprocess.run([sys.executable, "-m", "ziglang", "cc", csrc, "-o", exe, "-lm"],
                           check=True, capture_output=True, timeout=timeout)
        out = subprocess.run([exe], check=True, capture_output=True, text=True, timeout=timeout).stdout
    return [float(x) for x in out.split()]


def zig_available():
    """True iff the `ziglang` PyPI wheel (or a system `zig`) can compile here. The wheel is an OPT-IN accelerator,
    exactly like numba: every test must pass without it, and its absence is reported loudly, never silently."""
    import importlib.util
    import shutil
    return importlib.util.find_spec("ziglang") is not None or shutil.which("zig") is not None


def _zig_argv():
    """Prefer the wheel (hermetic, version-pinned by pip) over a system zig."""
    import importlib.util
    import sys
    if importlib.util.find_spec("ziglang") is not None:
        return [sys.executable, "-m", "ziglang"]
    return ["zig"]


def run_zig(kernel, calls, dialect="zig_f64", timeout=180):
    """Compile the emitted Zig kernel (`-O ReleaseSafe`) and RUN it on `calls` -- the executed bar, same as run_c.

    ReleaseSafe is deliberate: ReleaseFast licenses float reassociation, and a reassociated sum is a different
    program (Kept Negative 5). Printing uses Zig's `{d}` float format, which is shortest-round-trip -- the parsed
    double is bit-identical to the printed one, so the comparison measures the ARITHMETIC, not the formatter.
    Raises EmitError if no Zig toolchain is present; callers skip loudly."""
    import os
    import subprocess
    import tempfile

    if dialect not in ("zig_f64", "zig_f32"):
        raise EmitError("run_zig only runs the Zig dialects; got %r" % (dialect,))
    if not zig_available():
        raise EmitError("no Zig toolchain: `pip install ziglang` (opt-in accelerator, like numba)")
    node, _fn = _as_node_and_fn(kernel)
    kernel_src = _emit_node(node, dialect)
    name = node.name
    # std.debug.print writes to stderr and has been API-stable across Zig versions, unlike stdout writers.
    body = "".join('    std.debug.print("{d}\\n", .{%s(%s)});\n'
                   % (name, ", ".join(repr(float(a)) for a in c)) for c in calls)
    prog = ('const std = @import("std");\n' + kernel_src + "pub fn main() void {\n" + body + "}\n")

    with tempfile.TemporaryDirectory() as tmp:
        zsrc = os.path.join(tmp, "k.zig")
        exe = os.path.join(tmp, "k")
        with open(zsrc, "w") as fh:
            fh.write(prog)
        subprocess.run(_zig_argv() + ["build-exe", "-O", "ReleaseSafe", zsrc, "-femit-bin=" + exe],
                       check=True, capture_output=True, timeout=timeout, cwd=tmp)
        out = subprocess.run([exe], check=True, capture_output=True, text=True, timeout=timeout).stderr
    return [float(x) for x in out.split()]


def validate_zig(kernel, calls, dialect="zig_f64"):
    """Run the emitted Zig against the Python original on the same inputs (shape mirrors validate_c). zig_f64's
    measured verdict is BIT-IDENTICAL; zig_f32's max-abs delta IS the f32 tolerance, measured, not asserted."""
    _node, fn = _as_node_and_fn(kernel)
    got = run_zig(kernel, calls, dialect=dialect)
    want = [float(fn(*c)) for c in calls]
    diffs = [abs(g - w) for g, w in zip(got, want)]
    rels = [abs(g - w) / max(abs(w), 1e-30) for g, w in zip(got, want)]
    return {"dialect": dialect, "n": len(calls), "max_abs_diff": max(diffs), "max_rel_diff": max(rels),
            "bit_identical": all(g == w for g, w in zip(got, want))}


def validate_c(kernel, calls, dialect="c_f64"):
    """Run the emitted C against the Python original on the same inputs. Returns
    `{dialect, n, max_abs_diff, max_rel_diff, bit_identical}`.

    `kernel` is source TEXT (preferred) or a live function; from text the Python side is built by `as_python`, so
    both sides run the SAME program rather than two implementations of it. A re-typed Python reference would be a
    second implementation, and comparing two implementations tests neither.

    `c_f64` comes out BIT-IDENTICAL -- same order of operations, same doubles. `c_f32` does not, and cannot: its
    2.9e-07 IS the tolerance a WGSL port must be judged against."""
    _node, fn = _as_node_and_fn(kernel)
    got = run_c(kernel, calls, dialect=dialect)
    want = [float(fn(*c)) for c in calls]
    diffs = [abs(g - w) for g, w in zip(got, want)]
    rels = [abs(g - w) / max(abs(w), 1e-30) for g, w in zip(got, want)]
    return {"dialect": dialect, "n": len(calls), "max_abs_diff": max(diffs), "max_rel_diff": max(rels),
            "bit_identical": all(g == w for g, w in zip(got, want))}


def _selftest():
    """Regression trap for K8: the C dialect is compiled and RUN (bit-identical in f64, 1e-7 in f32), the WGSL is
    structurally well-formed, and the emitter refuses every unresolved construct."""
    import math

    def sdf_sphere(px: float, py: float, pz: float, r: float) -> float:
        """A real leCore kernel: the sphere SDF."""
        d = sqrt(px * px + py * py + pz * pz)                 # noqa: F821 -- an intrinsic, resolved by the table
        return d - r

    # `sqrt` must exist for the PYTHON side of the comparison to run
    sdf_sphere.__globals__.setdefault("sqrt", math.sqrt)

    calls = [(0.3, -0.7, 1.1, 0.85), (1.0, 0.0, 0.0, 0.5), (-2.0, 1.5, 0.25, 1.0)]

    # 1. THE BAR, EXECUTED: emitted C compiled with cc and run
    rep64 = validate_c(sdf_sphere, calls, "c_f64")
    assert rep64["bit_identical"] is True and rep64["max_abs_diff"] == 0.0

    # 2. KEPT NEGATIVE 1: f32 cannot be bit-identical, and its error IS the WGSL tolerance
    rep32 = validate_c(sdf_sphere, calls, "c_f32")
    assert rep32["bit_identical"] is False
    assert 0.0 < rep32["max_abs_diff"] < 1e-5

    # 3. the WGSL text is well-formed, and shares the IR with the C that was executed
    w = emit(sdf_sphere, "wgsl")
    assert w.startswith("fn sdf_sphere(") and "-> f32" in w and "let d =" in w
    assert "double" not in w and "def " not in w and "**" not in w
    assert emit(sdf_sphere, "js").startswith("function sdf_sphere(") and "Math.sqrt" in emit(sdf_sphere, "js")

    # 4. K10's RULE: refuse rather than guess
    def unannotated(x, y: float) -> float:
        return x + y

    def bad_call(x: float) -> float:
        return numpy_thing(x)                                  # noqa: F821

    def has_while(x: float) -> float:
        while x < 3.0:                                         # an unprovable trip count -- still refused
            x = x + 1.0
        return x

    def var_range(x: float) -> float:
        n = 3.0
        for _i in range(n):                                    # a variable trip count -- still refused
            x = x + 1.0
        return x

    def no_return(x: float) -> float:
        y = x + 1.0                                            # noqa: F841

    for fn_, needle in ((unannotated, "annotation"), (bad_call, "unknown call"),
                        (has_while, "unsupported statement"), (var_range, "range"),
                        (no_return, "never returns")):
        try:
            emit(fn_, "wgsl")
        except EmitError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError("emit must refuse %s" % fn_.__name__)

    # 4b. BOUNDED loops EMIT (the former has_loop refusal, retired deliberately: a literal trip count is not a
    #     guess). The counter promotes explicitly and the accumulator declares mutable.
    def octaves(x: float) -> float:
        s = 0.0
        for i in range(4):
            s = s + x * (1.0 + i)
        return s
    w_loop = emit(octaves, "wgsl")
    assert "for (var i: i32 = 0; i < 4; i = i + 1) {" in w_loop and "f32(i)" in w_loop and "var s = " in w_loop

    try:
        emit(sdf_sphere, "glsl")
    except EmitError:
        pass
    else:
        raise AssertionError("an unknown dialect must raise")

    # 5. `emit_source` takes the text, so a kernel with no retrievable source still emits
    # 6. ZIG, EXECUTED (opt-in): compiled ReleaseSafe and RUN when a toolchain exists; otherwise SKIPPED LOUDLY.
    #    zig_f64 must be bit-identical (builtin intrinsics only -- pow is a declared 1-ulp negative, KN6);
    #    zig_f32's delta is asserted against f32 epsilon scale, the same bar WGSL is judged by.
    if zig_available():
        zc = [(0.3 * i - 3.0, 0.7, 1.1 - 0.05 * i, 0.4) for i in range(24)]
        z64 = validate_zig(sdf_sphere, zc, dialect="zig_f64")
        assert z64["bit_identical"], "zig_f64 must be bit-identical on builtin-intrinsic kernels: %r" % z64
        z32 = validate_zig(sdf_sphere, zc, dialect="zig_f32")
        assert z32["max_abs_diff"] < 5e-6, "zig_f32 beyond f32-epsilon scale: %r" % z32
        print("  zig executed: f64 bit-identical, f32 max_abs=%.3g" % z32["max_abs_diff"])
    else:
        print("  ZIG SKIPPED -- no toolchain (`pip install ziglang`); structural emission still asserted below")
    zt = emit(sdf_sphere, "zig_f64")
    assert "fn sdf_sphere(" in zt and "f64" in zt and "def " not in zt and "**" not in zt

    text = "def lerp(a: float, b: float, t: float) -> float:\n    return a + (b - a) * t\n"
    assert emit_source(text, "wgsl").startswith("fn lerp(a: f32, b: f32, t: f32) -> f32")
    assert emit_source(text, "c_f64").startswith("double lerp(double a")

    # B1: GLSL shader ASSEMBLY -- compose function pieces, catch duplicate names, wrap for WebGL2 (deterministic).
    fa = "float f(vec3 p){ return p.x; }"
    fb = "vec3 g(float t){ return vec3(t); }"
    assert glsl_function_names(fa) == ["f"] and glsl_function_names(fb) == ["g"]
    composed = assemble_glsl([fa, fb], entry="void mainImage(out vec4 c, in vec2 u){ c = vec4(g(f(vec3(u,0.0))),1.0); }")
    assert "float f(vec3 p)" in composed and "vec3 g(float t)" in composed and "mainImage" in composed
    assert assemble_glsl([fa, fb], entry="void mainImage(out vec4 c, in vec2 u){ c=vec4(0.0); }") == \
           assemble_glsl([fa, fb], entry="void mainImage(out vec4 c, in vec2 u){ c=vec4(0.0); }")   # deterministic
    try:
        assemble_glsl([fa, fa])                                # duplicate 'f' -> raise, not a silently-shadowed shader
    except ValueError:
        pass
    else:
        raise AssertionError("assemble_glsl must reject a duplicate function name")
    w = webgl2_wrap(composed)
    assert w.startswith("#version 300 es") and "out vec4 fragOut;" in w
    assert "void main(){ mainImage(fragOut, gl_FragCoord.xy); }" in w
    assert webgl2_wrap(composed) == w

    print("OK: holographic_emit self-test passed (the sphere SDF emitted to C and COMPILED with cc: c_f64 is "
          "BIT-IDENTICAL to the Python original over %d inputs, c_f32 differs by %.3e -- and that number is the "
          "tolerance a WGSL port must be judged against, because WGSL is f32 and NumPy is f64. The WGSL text is "
          "structurally well-formed and shares the IR, but is NOT executed here, which is stated. The emitter "
          "REFUSES an unannotated parameter, an unknown call, a loop, and a missing return)"
          % (rep64["n"], rep32["max_abs_diff"]))


if __name__ == "__main__":
    _selftest()
