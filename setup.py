# setup.py -- packaging for leCore.
#
# leCore is a flat collection of ~260 `holographic_*.py` modules that import each other by their plain
# top-level names (e.g. `from holographic_ai import bind`). The simplest, least-surprising way to ship
# that is to install them AS top-level modules, exactly as they sit in the repo -- so anything that works
# from a clone works once installed, with NO import rewrites and NO sys.path tricks.
#
# We glob the module list at build time (so adding a new holographic_*.py needs no edit here) and add one
# small convenience module, `lecore.py`, that re-exports the main API -- so callers can just `import lecore`.

import glob
import os
from setuptools import setup

here = os.path.dirname(os.path.abspath(__file__))

def read(name):
    path = os.path.join(here, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""

# every holographic_*.py in this folder becomes a top-level module in the wheel.
# (test files are named test_holographic_*.py, so this glob already excludes them; the build script
#  also copies only the non-test modules into the staging folder, belt-and-suspenders.)
modules = [
    os.path.splitext(os.path.basename(path))[0]
    for path in sorted(glob.glob(os.path.join(here, "holographic_*.py")))
    if not os.path.basename(path).startswith("test_")
]
modules.append("lecore")                 # the convenience shim: `import lecore` -> lecore.UnifiedMind

setup(
    name="lecore",
    version="0.1.0",                     # bump per release (or let CI set it from the git tag -- see PACKAGING.md)
    description="leCore -- the vector-symbolic core of leOS: memory, geometry, physics and more on one NumPy substrate.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="AnOversizedMooseWithSocks",
    url="https://github.com/AnOversizedMooseWithSocks/leCore",
    py_modules=modules,                  # <- install these flat, top-level modules (no package nesting)
    python_requires=">=3.9",
    install_requires=["numpy"],          # the core needs ONLY NumPy -- nothing else is ever required
    extras_require={                      # opt-in extras -- the core runs, and passes every test, without them.
        # Install one with:  pip install .[jit]     (from the cloned folder -- note the dot)
        # or several:        pip install .[ui,jit]
        #
        # -- optional accelerators --
        "jit":      ["numba"],            # numba-compiled fast paths (holographic_jit / sdf_render / codegen)
        "symbolic": ["sympy"],            # design-time symbolic gradients (holographic_codegen / sdf_render)
        "gpu":      ["cupy"],             # GPU backend (holographic_backend). NOTE: CuPy is tied to your CUDA
                                          #   version -- you often need a specific wheel like `cupy-cuda12x`
                                          #   instead, so it is best installed by hand (and left out of `all`).
        # -- optional tooling --
        "ui":       ["flask", "pillow"],  # the browser UI (app.py) + image load/save
        "dev":      ["pytest", "matplotlib"],   # run the test suite and generate the plots
        # -- convenience: everything portable in one shot (CuPy excluded -- see the note above) --
        "all":      ["numba", "sympy", "flask", "pillow", "pytest", "matplotlib"],
    },
)
