#!/usr/bin/env python3
"""lint_scripts.py -- catch the bugs that crash a 90-minute run in the first 0.4 seconds. stdlib only.

WHY THIS EXISTS
`distill_router.py` did this:

    import lecore_paths as P          # module alias
    ...
    def main():
        a.weights = a.weights or str(P.nomic_weights())    # line 182
        ...
        P = Vt[:d].T                                        # line 218 -- a projection matrix

Python scopes a name locally if it is assigned ANYWHERE in the function body, so `P` at line 182 is
the *local* that will be assigned at line 218: `UnboundLocalError`. The stage died in 0.4s, after the
driver had already been asked to spend 30 minutes on it.

Nothing about that bug needed the weights, or numpy, or a single second of compute to find. It is a
pure AST property. So: check it, and the other cheap crash-in-the-first-second properties, before any
long stage runs. This is the same principle as the reachability audit -- a static check whose whole
value is that it runs before the expensive thing.

CHECKS, BY SEVERITY -- and the severity is the whole design.
An ERROR guarantees a crash. A WARN is advice. Only ERRORs may halt a 90-minute run; the first version
halted on advice, which is the same crying-wolf failure the order-blind shadowing check had. A lint
that stops work over a style note gets disabled, and then it catches nothing.

  ERROR [1] module-alias shadowing   reads `alias.attr` before assigning `alias` -> UnboundLocalError
  ERROR [4] syntax                   the file will not compile
  WARN  [2] module-scope name        used but never bound (approximate; function scope not inferred)
  WARN  [3] required positional      only for scripts `run_all.py` invokes bare (they import lecore_paths)

Scope note for [3]: `codebase_index.py`, `codec_probe.py`, `semantic_bridge.py` etc. are standalone
tools that are MEANT to take arguments. They are not run_all stages, so a required positional is
correct there and the lint must not say otherwise. The tell is `import lecore_paths` -- a script that
self-resolves its paths is one run_all calls bare.

USAGE
    python3 lint_scripts.py            # lint every .py in scripts/
    python3 lint_scripts.py --strict   # warnings become errors
    python3 lint_scripts.py foo.py     # or just one
"""
import ast, pathlib, sys


def module_aliases(tree):
    """Names bound by `import x as A` / `from m import y as A` / `import x`."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                out[al.asname or al.name.split('.')[0]] = al.name
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                out[al.asname or al.name] = f"{node.module}.{al.name}"
    return out


def assigned_names(fn):
    """Every name the function body BINDS -- which is what makes it local for the whole body."""
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        out.add(n.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            out.add(node.name)
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            out.add(node.optional_vars.id)
    # a `global X` declaration means the assignment is NOT local -- no shadowing
    for node in ast.walk(fn):
        if isinstance(node, ast.Global):
            out -= set(node.names)
    return out


def _store_linenos(fn, name):
    return [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)]


def _attr_read_linenos(fn, name):
    return [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == name and isinstance(n.value.ctx, ast.Load)]


def check_shadowing(path, tree, aliases):
    """[1] The bug that cost us a run. A function that assigns an imported alias makes that name LOCAL
    for the whole body -- so any earlier `alias.attr` read raises UnboundLocalError.

    ORDER MATTERS, and the first version of this lint ignored it: `core = np.asarray(...)` followed by
    `core.size` is perfectly legal (assignment precedes use), and flagging it is crying wolf. A lint that
    cries wolf gets ignored, which is worse than no lint. So: report only when the FIRST attribute read
    precedes the FIRST assignment.

    (Still approximate -- a read inside a loop that runs after the assignment is fine, and a read in a
     branch that runs before it is not. Line order is the honest cheap proxy; it caught the real bug and
     it clears the real scripts.)"""
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for name in sorted(assigned_names(fn) & set(aliases)):
            reads, stores = _attr_read_linenos(fn, name), _store_linenos(fn, name)
            if reads and stores and min(reads) < min(stores):
                bad.append(f"{path.name}:{min(reads)} in def {fn.name}(): reads `{name}.…` at line "
                           f"{min(reads)} but assigns `{name}` at line {min(stores)}; `{name}` is "
                           f"`import {aliases[name]}` -- UnboundLocalError. Rename one of them.")
    return bad


def check_required_positionals(path, tree, aliases):
    """[3] run_all.py calls its stages with no arguments, so a stage may not have a required positional.
    But most scripts here are standalone tools that SHOULD take arguments. Only check the ones that
    import `lecore_paths` -- i.e. the ones that advertise "I resolve my own paths"."""
    if 'lecore_paths' not in ' '.join(aliases.values()):
        return []
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if not isinstance(name, str) or name.startswith('-'):
            continue
        kw = {k.arg for k in node.keywords}
        if 'nargs' not in kw and 'default' not in kw:
            bad.append(f"{path.name}:{node.lineno} add_argument({name!r}) is a REQUIRED positional; "
                       f"add nargs='?' + a default so `run_all.py` can call it bare.")
    return bad


def check_module_scope_names(path, tree, aliases):
    """[2] A cheap undefined-name check at module scope only (function scope needs real inference)."""
    bound = set(aliases) | {'__name__', '__file__', '__doc__'}
    # Names bound anywhere at module level -- INCLUDING inside `if __name__ == '__main__':` blocks,
    # which the first version missed and then reported `V` as undefined. Same wolf-crying problem.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)                 # `except X as e` binds e (false positive #4)
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            bound.add(node.optional_vars.id)     # `with x as f` binds f
    bad = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound \
                    and n.id not in dir(__builtins__):
                bad.append(f"{path.name}:{n.lineno} module-scope name `{n.id}` is never bound")
    return bad


def lint(path):
    """Returns (errors, warnings)."""
    src = path.read_text(encoding='utf-8', errors='replace')
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [f"{path.name}:{e.lineno} SYNTAX: {e.msg}"], []
    aliases = module_aliases(tree)
    errors = check_shadowing(path, tree, aliases)
    warns = check_module_scope_names(path, tree, aliases) + check_required_positionals(path, tree, aliases)
    return errors, warns


def main():
    here = pathlib.Path(__file__).resolve().parent
    argv = [a for a in sys.argv[1:] if not a.startswith('-')]
    strict = '--strict' in sys.argv
    files = [pathlib.Path(p) for p in argv] or sorted(here.glob('*.py'))
    errors, warns = [], []
    for f in files:
        if f.name == pathlib.Path(__file__).name:
            continue
        e, w = lint(f)
        errors += e; warns += w
    if strict:
        errors, warns = errors + warns, []
    for w in warns:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print(f"\nlint_scripts: {len(files)} file(s) | {len(errors)} error(s), {len(warns)} warning(s)")
    if errors:
        print("errors guarantee a crash -- fix before the long stages run.")
        sys.exit(1)
    print("no errors; safe to run.")


if __name__ == '__main__':
    main()
