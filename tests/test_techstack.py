"""Tech-stack regression guards -- turn the one-time 'are we using the correct stack' sweep into permanent checks.

Two properties the project's constitution promises, enforced so they can't silently rot:
  1. The CORE runs on NumPy alone. Importing the mind and using a faculty must NOT require any optional/banned dep
     (PIL, matplotlib, nltk, scipy, sklearn, torch, cupy, numba). If a future change leaks one of those into a
     core-reachable import, this fails. Run in a SUBPROCESS so blocking those modules can't disturb the test session.
  2. No module wires a SUPERSEDED duplicate. The 4 query_* modules were replaced by wired twins (querylock/querytime/
     queryprog/querygraph) and carry a 'SUPERSEDED BY' banner; nothing but their own module+test should import them,
     so a duplicate implementation can never quietly come back into a pipeline.
"""
import os
import sys
import glob
import subprocess


def test_core_imports_on_numpy_alone():
    """A clean interpreter that BLOCKS every optional/banned dep must still import lecore + UnifiedMind and run a
    faculty. This is the 'core needs only NumPy' guarantee, checked the hard way."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys\n"
        "for banned in ('PIL','matplotlib','nltk','scipy','sklearn','torch','cupy','numba','pyfftw','sympy'):\n"
        "    sys.modules[banned] = None            # any 'import <banned>' now raises ImportError\n"
        "import numpy\n"
        "import lecore\n"
        "from holographic.misc.holographic_unified import UnifiedMind\n"
        "m = UnifiedMind(dim=256, seed=0)\n"
        "assert m.find_capability('search a big pile of vectors')\n"
        "assert m.build_scene('a red sphere').objects\n"
        "print('CORE_PURE_OK')\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, env=env, timeout=180)
    assert "CORE_PURE_OK" in r.stdout, "core is not NumPy-only:\n%s\n%s" % (r.stdout, r.stderr)


def test_no_module_imports_a_superseded_module():
    """Nothing (outside a superseded module's own file + its own test) may import it -- otherwise a duplicate is
    creeping back into a pipeline instead of using the wired twin."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    superseded = ["holographic_query_concurrency", "holographic_query_history",
                  "holographic_query_programs", "holographic_query_graph"]
    offenders = {}
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        base = os.path.basename(path)[:-3]
        if base.startswith("test_"):
            continue
        # the module itself, the reachability audit, and the buried-audit are allowed to name them
        if base in ("tools",):
            continue
        try:
            txt = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for sup in superseded:
            if sup == base:
                continue
            # a real import, not just a mention in a comment/docstring
            if ("import %s" % sup) in txt or ("from %s " % sup) in txt:
                offenders.setdefault(base, []).append(sup)
    assert not offenders, "modules importing a SUPERSEDED twin (use the wired version instead): %s" % offenders


def test_superseded_modules_still_carry_their_banner():
    """The 4 duplicates must keep their 'SUPERSEDED BY' banner so a reader is redirected to the wired twin."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for m in ("holographic_query_concurrency", "holographic_query_history",
              "holographic_query_programs", "holographic_query_graph"):
        matches = glob.glob(os.path.join(root, "holographic", "**", m + ".py"), recursive=True)
        assert matches, "could not find %s.py anywhere under holographic/" % m
        txt = open(matches[0], encoding="utf-8").read()
        assert "SUPERSEDED BY" in txt, "%s lost its superseded banner" % m


def test_every_optional_import_has_a_declared_extra():
    """Every third-party package the engine imports OPTIONALLY (numpy is the only hard dep) must be installable via
    a `pip install leos-core[extra]` -- otherwise a user hits `ImportError: No module named X` on an accelerated
    path with no documented way to get X. This pins the setup.py extras against the actual imports, so a newly
    added optional dependency (like pyfftw was) can't ship without an extra. Import name -> distribution name is
    mapped where they differ (PIL->pillow)."""
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # the packages the CORE is allowed to hard-import (no extra needed) + pure stdlib/first-party
    ALWAYS_OK = {"numpy"}
    # import name -> the pip distribution name that provides it, when they differ
    DIST = {"PIL": "pillow"}
    # third-party import names we care about (accelerators + optional tooling); everything else is stdlib/first-party
    OPTIONAL_IMPORTS = {"numba", "cupy", "sympy", "pyfftw", "PIL", "matplotlib", "nltk", "flask"}

    # 1. scan the engine for which of those it actually imports
    found = set()
    for path in glob.glob(os.path.join(root, "holographic", "**", "*.py"), recursive=True):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    if top in OPTIONAL_IMPORTS:
                        found.add(top)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in OPTIONAL_IMPORTS:
                    found.add(top)

    # 2. read the distributions setup.py's extras can install
    setup_src = open(os.path.join(root, "setup.py"), encoding="utf-8").read()
    tree = ast.parse(setup_src)
    extras = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for kw in node.keywords:
                if kw.arg == "extras_require":
                    extras = ast.literal_eval(kw.value)
    installable = {pkg.lower() for pkgs in extras.values() for pkg in pkgs}

    # 3. every optional import must be reachable through some extra (by its DIST name)
    missing = []
    for imp in sorted(found):
        dist = DIST.get(imp, imp).lower()
        if dist not in installable and dist not in ALWAYS_OK:
            missing.append("%s (needs extra providing %r)" % (imp, dist))
    assert not missing, (
        "optional imports with NO installable extra -- add them to setup.py's extras_require:\n  "
        + "\n  ".join(missing))


def test_all_extra_is_the_union_minus_cupy():
    """`all` is the convenience 'everything portable' extra: it must equal the union of the other extras minus cupy
    (CuPy is CUDA-version-specific, installed by hand). If someone adds an extra and forgets to fold it into `all`,
    `pip install leos-core[all]` silently misses it -- this catches that."""
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(root, "setup.py"), encoding="utf-8").read())
    extras = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for kw in node.keywords:
                if kw.arg == "extras_require":
                    extras = ast.literal_eval(kw.value)

    union = set()
    for name, pkgs in extras.items():
        if name == "all":
            continue
        union.update(pkgs)
    union.discard("cupy")                    # deliberately excluded from all
    assert set(extras["all"]) == union, (
        "`all` is out of sync with the other extras (minus cupy).\n"
        "  in all but not union: %s\n  in union but not all: %s"
        % (sorted(set(extras["all"]) - union), sorted(union - set(extras["all"]))))
