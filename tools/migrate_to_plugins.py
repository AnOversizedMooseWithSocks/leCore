"""Move optional-dependency faculties out of the UnifiedMind parts into bundled plugins.

Mechanical on purpose: the body of each faculty is copied VERBATIM (minus `self`), so
the migration is a move and not a rewrite -- a rewrite hidden inside a migration is how
a refactor silently changes results. Each plugin's PLUGIN dict names the pip extra that
installs its dependency, so `pip install leos-core[zig]` and the `zig` plugin are one
thing under two names.
"""
import inspect
import importlib
import textwrap
import os

import lecore

PLAN = {
    "zig": {
        "verbs": ["zig_batch_eval", "zig_regime_map", "zig_dispatch_policy",
                  "zig_march_compare", "validate_kernel"],
        "requires": ("ziglang",), "install": "pip install leos-core[zig]",
        "does": "Native Zig kernels: compile scalar kernels to shared libraries, race them "
                "against NumPy, and validate emitted C/Zig against the Python original.",
        "why": "Every verb here exists to drive the `ziglang` wheel (a ~45 MB toolchain). "
               "Without it each already answered honestly ('ziglang not installed'); that "
               "answer is unchanged, the verbs just no longer live on the core class.",
    },
    "symbolic": {
        "verbs": ["compiled_sdf_normal", "exact_sdf_normal", "gradient_cache_symbolic"],
        "requires": ("sympy",), "install": "pip install leos-core[symbolic]",
        "does": "Design-time symbolic maths: exact SDF normals and Jacobians derived with "
                "SymPy and compiled once into the content-addressed cache.",
        "why": "SymPy is a design-time dependency: these derive an exact expression ONCE and "
               "hand a plain callable to the renderer. The renderer itself stays in core and "
               "never imports sympy.",
    },
    "jit": {
        "verbs": ["compiled_sdf_numba", "render_sdf_fast"],
        "requires": ("numba", "sympy"), "install": "pip install leos-core[jit,symbolic]",
        "does": "Numba-compiled SDF kernels and the fully-JIT'd analytic SDF renderer "
                "(SymPy expression in, njit kernels out).",
        "why": "These need BOTH numba and sympy and exist only to produce JIT'd kernels. "
               "NOT moved: signed_distance_field and every other faculty with a numba FAST "
               "PATH and a NumPy fallback -- that is the accelerator pattern, and it stays in "
               "core precisely because it works without numba.",
    },
    "wgsl": {
        "verbs": ["run_wgsl_kernel"],
        "requires": ("wgpu",), "install": "pip install leos-core[wgsl]",
        "does": "Run an annotated Python kernel on any GPU (Vulkan / Metal / DX12 / WebGPU, "
                "or a software adapter) via its WGSL projection.",
        "why": "The WGSL EMITTER stays in core (it is pure Python and used by the dialect "
               "emitters); only RUNNING the result needs the wgpu wheel.",
    },
    "gpu": {
        "verbs": ["unicron_device"],
        "requires": ("cupy",), "install": "pip install cupy-cuda12x  (match your CUDA)",
        "does": "Run the language model on the CuPy/CUDA backend and prove it agrees with "
                "the NumPy path.",
        "why": "CuPy is tied to the host's CUDA version and is deliberately left out of the "
               "`all` extra; a verb that only makes sense with it is the definition of a "
               "plugin. The transparent CuPy array backend (holographic_backend, "
               "mind.use_gpu) stays in core because it has a NumPy fallback.",
    },
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repo root


def _find_part(name):
    """Which unified part defines this faculty, and its class."""
    for f in sorted(os.listdir(os.path.join(ROOT, "holographic", "unified"))):
        if not f.startswith("holographic_unified_p") or not f.endswith(".py"):
            continue
        modname = "holographic.unified." + f[:-3]
        mod = importlib.import_module(modname)
        for cn, cls in vars(mod).items():
            if inspect.isclass(cls) and cls.__module__ == modname and name in vars(cls):
                return os.path.join(ROOT, "holographic", "unified", f), cls, cn
    raise KeyError(name)


def _to_function(method_src):
    """A method body -> a module-level function: dedent, drop `self`."""
    src = textwrap.dedent(method_src)
    head, rest = src.split("\n", 1)
    head = head.replace("(self, ", "(").replace("(self)", "()")
    return head + "\n" + rest


def _method_source(part_path, cls, name):
    """The method's source read from the CURRENT file text, not inspect's cache.

    THE BUG THIS FIXES (it bit on the first run): inspect.getsource reads through
    linecache, which is populated at import. Rewriting part 11 for the `symbolic` plugin
    left the cache stale, so the `jit` step -- same file, later -- got the source of the
    function that NOW sat at the old line numbers: `compile_program` came back under
    the name `compiled_sdf_numba`, and the wrong method was removed from core. A move
    that reads stale text is a silent corruption, so every source is read up front from
    the file as it is on disk, before anything is rewritten.
    """
    import linecache
    linecache.checkcache(part_path)
    text = open(part_path, encoding="utf-8").read()
    header = "    def %s(self" % name
    start = text.index(header)
    # the method ends where the next line at class-body indentation begins
    lines = text[start:].split("\n")
    body = [lines[0]]
    for ln in lines[1:]:
        if ln and not ln.startswith("        ") and not ln.startswith("    #") and ln.strip():
            break
        body.append(ln)
    # trim trailing blank lines so removal is exact
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body) + "\n"


def main():
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m.set_file_root(ROOT)

    # PHASE 1 -- extract every source from the files as they are on disk. Nothing is
    # written until every read is done.
    extracted = {}                       # (plugin, verb) -> (part_path, src, doc, sig)
    for pname, spec in PLAN.items():
        for vname in spec["verbs"]:
            part_path, cls, cn = _find_part(vname)
            fn = getattr(cls, vname)
            src = _method_source(part_path, cls, vname)
            assert src.lstrip().startswith("def %s(self" % vname), src[:80]
            doc = (inspect.getdoc(fn) or "").strip().split("\n")[0]
            try:
                sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
            except (TypeError, ValueError):
                sig = "(...)"
            extracted[(pname, vname)] = (part_path, src, doc, sig)

    # PHASE 2 -- write the plugins, then remove from the parts.
    for pname, spec in PLAN.items():
        funcs, verbs, removed = [], [], {}
        for vname in spec["verbs"]:
            part_path, src, doc, sig = extracted[(pname, vname)]
            funcs.append(_to_function(src))
            verbs.append((vname, doc, sig))
            removed.setdefault(part_path, []).append(src)

        out = ['"""%s -- bundled plugin (pip extra: %s).\n\n%s\n\nMOVED VERBATIM from the UnifiedMind parts by tools/migrate_to_plugins.py: bodies and\n'
               'docstrings are the former methods with `self` dropped, so GET /tools, find_capability and\n'
               'REFERENCE.md say what they said before. A move, not a rewrite.\n"""\n'
               % (pname.upper(), spec["install"], spec["why"]),
               "PLUGIN = {\n    \"name\": %r,\n    \"version\": \"1.0\",\n    \"does\": %r,\n    \"requires\": %r,\n    \"install\": %r,\n}\n"
               % (pname, spec["does"], tuple(spec["requires"]), spec["install"])]
        out.extend("\n\n" + f for f in funcs)
        out.append("\n\ndef register(mind, config=None):\n"
                   "    \"\"\"Bind this plugin's verbs. `config` is accepted and unused.\"\"\"\n"
                   "    return [\n")
        for vname, doc, sig in verbs:
            out.append("        {\"name\": %r, \"fn\": %s,\n         \"does\": %r,\n         \"example\": %r},\n"
                       % (vname, vname, doc[:200], "mind.%s%s" % (vname, sig)))
        out.append("    ]\n")
        out.append("\n\ndef _selftest():\n"
                   "    \"\"\"Contract: the verbs bind on a default mind and are absent from a slim one; each\n"
                   "    is callable (its own honest failure without the dependency is the fallback that\n"
                   "    made this a plugin in the first place).\"\"\"\n"
                   "    import lecore\n"
                   "    m = lecore.UnifiedMind(dim=64, seed=0)\n"
                   "    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())\n"
                   "    for v in %r:\n"
                   "        assert callable(getattr(m, v, None)), 'bundled %s did not bind %%r' %% v\n"
                   "        assert not hasattr(slim, v), 'plugins=() still carried %%r' %% v\n"
                   "    assert [p['name'] for p in m.plugin_list() if p['name'] == %r] == [%r]\n"
                   "    print('holographic.plugins.%s selftest OK')\n\n\n"
                   "if __name__ == \"__main__\":\n    _selftest()\n"
                   % (spec["verbs"], pname, pname, pname, pname))
        dest = os.path.join(ROOT, "holographic", "plugins", pname + ".py")
        open(dest, "w", encoding="utf-8").write("".join(out))
        print("wrote", dest, "chk", m.file_python_check(dest))

        for part_path, srcs in removed.items():
            text = open(part_path, encoding="utf-8").read()
            for s in srcs:
                assert s in text, "could not locate verbatim in %s" % part_path
                text = text.replace(s + "\n", "", 1) if (s + "\n") in text else text.replace(s, "", 1)
            open(part_path, "w", encoding="utf-8").write(text)
            rel = os.path.relpath(part_path, ROOT)
            print("  removed %d from %s chk %s" % (len(srcs), rel, m.file_python_check(rel)))


if __name__ == "__main__":
    main()
