# setup.py -- packaging for leCore.
#
# leCore's engine code lives in the `holographic` package (holographic/<family>/holographic_*.py), imported as
# e.g. `from holographic.rendering.holographic_camera import ...`. We ship it as a real package tree via
# find_packages() (so adding a new module/subpackage needs no edit here), plus one small top-level convenience
# module, `lecore.py`, that re-exports the main API -- so callers can just `import lecore`.

import os
from setuptools import setup, find_packages

here = os.path.dirname(os.path.abspath(__file__))

def read(name):
    path = os.path.join(here, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""

# every package under holographic/ (holographic itself + every family subpackage) ships in the wheel.
# lecore_data is a separate runtime-data package, declared alongside it below.
engine_packages = find_packages(where=here, include=["holographic", "holographic.*"])

setup(
    # NOTE ON THE NAME: the *distribution* name (what you `pip install`) and the *import* name (what you
    # `import` in Python) are independent. The plain name "lecore" is already taken on PyPI by an unrelated
    # project, and this engine is the core of the larger leOS project -- so we publish as "leos-core" but the
    # modules still install at the top level, so users write `import lecore` (via the lecore.py shim) exactly
    # as they did from a clone. Install:  pip install leos-core   ->   then:  import lecore
    name="leos-core",
    version="0.1.0",                     # bump per release (or let CI set it from the git tag -- see PACKAGING.md)
    description="leOS-core (import name: lecore) -- the vector-symbolic core of leOS: memory, geometry, physics and more on one NumPy substrate.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="AnOversizedMooseWithSocks",
    url="https://github.com/AnOversizedMooseWithSocks/leCore",
    py_modules=["lecore", "holographic_service", "holographic_product", "holographic_x402_api"],
    packages=engine_packages + ["lecore_data"],   # <- the real holographic/ package tree + the runtime data package
    # The runtime data (the WordNet dictionary, material property JSON) ships as the small `lecore_data` PACKAGE, so
    # it is carried into the wheel and resolves the same from a clone or an install (see lecore_data/__init__.py).
    include_package_data=True,
    package_data={
        "lecore_data": [
            "knowledge/*",                                  # dictionary.json.xz (lzma), manifest.json, LICENSE_WORDNET.txt
            "definitions/*.md",
            "definitions/native/materials/*.json",
            "definitions/standards/generic_table/*.json",
        ],
    },
    python_requires=">=3.9",
    install_requires=["numpy"],          # the core needs ONLY NumPy -- nothing else is ever required
    extras_require={                      # opt-in extras -- the core runs, and passes every test, without them.
        # Install one with:  pip install .[jit]     (from the cloned folder -- note the dot)
        # or several:        pip install .[ui,jit]
        #
        # -- optional accelerators --
        "jit":      ["numba"],            # numba-compiled fast paths (holographic_jit / sdf_render / codegen)
        "symbolic": ["sympy"],            # design-time symbolic gradients (holographic_codegen / sdf_render)
        "zig":      ["ziglang"],          # native batch kernels + raymarcher (holographic_zigrun / zigmarch):
                                          #   measured 2-5x over vectorised NumPy for repeated medium-n kernels,
                                          #   3.8x on the raymarch demo, BIT-IDENTICAL in safe mode. The wheel
                                          #   ships the whole Zig toolchain (~45 MB) -- no system compiler needed;
                                          #   it also backstops the C validation path via `zig cc`.
        "gpu":      ["cupy"],             # GPU backend (holographic_backend). NOTE: CuPy is tied to your CUDA
                                          #   version -- you often need a specific wheel like `cupy-cuda12x`
                                          #   instead, so it is best installed by hand (and left out of `all`).
        "x402":     ["x402[fastapi,evm]>=2.15,<3", "uvicorn>=0.51,<1"],  # paid API publishing
        # -- optional tooling --
        "ui":       ["flask", "pillow"],  # the browser UI (app.py) + image load/save
        "images":   ["pillow"],           # image I/O beyond stdlib PNG (jpg/webp/... via mind.save_render) --
                                          #   pillow without pulling in Flask; a subset of `ui` for headless use
        "dev":      ["pytest", "matplotlib"],   # run the test suite and generate the plots
        # -- convenience: everything portable in one shot (CuPy excluded -- see the note above) --
        "all":      ["numba", "sympy", "flask", "pillow", "pytest", "matplotlib", "ziglang"],
    },
)
