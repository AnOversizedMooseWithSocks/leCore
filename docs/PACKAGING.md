# Packaging leCore as an importable Python package

leCore ships as a normal pip-installable package. On PyPI the project is published as **`leos-core`** (the
plain name `lecore` was already taken by an unrelated project, and this engine is the core of the larger
**leOS** project). The important part: the *distribution* name and the *import* name are independent, so you
install one name and import another:

```sh
pip install leos-core        # the distribution name (what PyPI knows it as)
```
```python
import lecore                 # the import name is unchanged -- `import lecore` works from a clone and a wheel
import numpy as np
mind = lecore.UnifiedMind(dim=1024, seed=0)      # dim/seed are the first two args; see the class for the rest
# the raw VSA algebra is re-exported for convenience:
rng = np.random.default_rng(0)
c = lecore.bind(lecore.random_vector(1024, rng), lecore.random_vector(1024, rng))
```

The full engine is still importable directly for anything the shim doesn't re-export, e.g.
`from holographic.rendering.holographic_render import ...` — the engine ships as the `holographic` package
(grouped into families like `rendering`, `mesh_and_geometry`, `agents_and_reasoning`), so submodules are
addressed by their package path.

## How it's put together (and why)

The engine is ~399 `holographic_*.py` modules, grouped into the `holographic` package under family
subpackages (`rendering`, `mesh_and_geometry`, `agents_and_reasoning`, `scene_and_pipeline`, ...). They
import each other by package path (`from holographic.agents_and_reasoning.holographic_ai import bind`), and
the package installs **as that same tree**, unchanged — no import rewriting, no `sys.path` tricks. `lecore.py`
is a tiny top-level convenience shim that re-exports the main API so `import lecore` works like any package.
Three small files drive it:

- **`setup.py`** — uses `find_packages()` to ship the whole `holographic` package tree (adding a new module or
  subpackage needs no edit here), plus two top-level modules: the `lecore` shim and the standalone
  `holographic_service` HTTP server. Declares `numpy` as the only required dependency; everything else is an
  opt-in extra (see the table below): `jit`, `symbolic`, `gpu`, `ui`, `dev`, and `all`. It also ships the
  **`lecore_data`** package (below) via `package_data`, so the runtime data travels with the wheel.
- **`pyproject.toml`** — three lines telling `python -m build` to use setuptools.
- **`lecore.py`** — the friendly front door (`UnifiedMind` + the raw ops).
- **`lecore_data/`** — a small importable package holding the data the engine reads **at runtime** (the WordNet
  dictionary the `lookup` faculty uses; the material-property JSON the heat model uses). It is a *package* rather
  than a loose `data/` folder for one concrete reason: a bare `data/` directory is **not** reliably carried into
  a wheel, but a package (declared with `package_data`) is. `import lecore_data` resolves the same from a clone
  and an install, so the loaders (`holographic_dictionary`, `holographic_heat`) find their files either way. Demo-only
  datasets (market ticks, etc.) stay in the repo's `data/` folder and are deliberately left out of the wheel.

## Building it locally

```sh
pip install build            # one-time
sh build_package.sh          # copies the essentials to a clean folder, strips tests, builds the wheel
pip install dist/leos_core-*.whl
```

`build_package.sh` copies the `holographic_*.py` modules (no `test_*`, no app/tour/benchmarks) **and the
`lecore_data/` runtime-data package** into a clean `build_pkg/` folder, strips any `__pycache__`, and builds the
wheel from there — so the distribution has exactly the code and data it needs and nothing else. The release
workflow smoke-tests the built wheel in an isolated temp dir and asserts the bundled dictionary actually loads, so
a wheel that shipped without its data would fail the release rather than reach users.

## CI

`package.yml` goes in `.github/workflows/`. On every push to `main` and on version tags (`v*`) it builds the
wheel, **installs it in isolation and imports it as a smoke test**, uploads the wheel + sdist as build
artifacts, and — on a `v*` tag — attaches them to the GitHub Release. On a `v*` tag it **also publishes the
release to PyPI** (see below), so `pip install leos-core` picks it up.

> **If the "Attach to the GitHub Release" step fails with `Resource not accessible by integration`:** the job's
> token is read-only. Attaching files *writes* to the release (part of the `contents` scope), so the `build-wheel`
> job declares `permissions: contents: write`. That block is what grants it — don't remove it.

### Publishing to PyPI (trusted publishing — no stored token)

The `publish-pypi` job uses PyPI **Trusted Publishing** (OpenID Connect). GitHub proves its identity to PyPI
directly, so there is **no API token or password stored in the repo** — that's what the `id-token: write`
permission in the workflow is for. You do this one-time setup on PyPI, then every `v*` tag publishes itself:

1. Log in at <https://pypi.org>. (For a dry run first, do the same on <https://test.pypi.org>.)
2. Because `leos-core` doesn't exist on PyPI yet, add a **pending publisher**:
   Account menu → **Publishing** → "Add a new pending publisher". (After the first successful publish it
   becomes a normal publisher attached to the project.)
3. Fill in exactly:
   - **PyPI Project Name:** `leos-core`
   - **Owner:** `AnOversizedMooseWithSocks`
   - **Repository name:** `leCore`
   - **Workflow name:** `package.yml`
   - **Environment name:** leave blank (unless you uncomment the `environment: pypi` line in the workflow —
     if you set one there, set the same name here).
4. Save. Now cut a release: bump `version` in `setup.py`, commit, then push a tag:
   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```
   The workflow builds, smoke-tests, attaches the files to the GitHub Release, and publishes `leos-core 0.1.0`
   to PyPI. A version number can only be published **once** — to re-publish you must bump the version.

Reserving the parent name `leos` (optional): publishing `leos-core` does **not** reserve `leos`. If you want
the parent name held too, upload a small placeholder release under `name="leos"` (a separate project/repo, or
a one-off local `twine upload`), since PyPI only reserves the exact name you upload.

### Versioning
The version lives in `setup.py` (`version="0.1.0"`); bump it per release. To release, tag the matching version
(`git tag v0.1.0 && git push origin v0.1.0`).

CI guards the two staying in sync: on a tag push, the `package` workflow runs `python tools/check_version.py --expect
"$GITHUB_REF_NAME"` **before** building, and fails the release if the tag and `setup.py` disagree — so you can't
accidentally publish `0.1.0` under a `v0.2.0` tag (and PyPI never lets you re-upload a version, so catching it here
matters). You can run the same check locally: `python tools/check_version.py` prints the current version, and
`python tools/check_version.py --expect v0.2.0` tells you whether a tag you're about to push would match.

If you'd rather have the git tag *drive* the version instead of hand-bumping, add one line to the workflow before the
build step (this makes the check above always pass, since the tag becomes the source of truth):

```yaml
      - name: Set version from the tag (optional)
        if: startsWith(github.ref, 'refs/tags/v')
        run: sed -i "s/version=\"[^\"]*\"/version=\"${GITHUB_REF_NAME#v}\"/" setup.py
```

## Optional extras (opt-in dependencies)

The core requires **only NumPy**. Everything else is declared as a named "extra" in `setup.py`'s
`extras_require`, so a user opts in with `pip install .[name]` (from the cloned folder) or
`pip install leos-core[name]` (if installed from PyPI). Combine names with commas: `pip install .[ui,jit]`.

| Extra | Pulls in | For |
|---|---|---|
| `jit` | `numba` | numba-compiled fast paths (`holographic_jit`, `sdf_render`, `codegen`) |
| `symbolic` | `sympy` | design-time symbolic gradients (`holographic_codegen`, `sdf_render`) |
| `gpu` | `cupy` | the GPU backend (`holographic_backend`) — see the CuPy note |
| `ui` | `flask`, `pillow` | the browser UI (`app.py`) and image load/save |
| `dev` | `pytest`, `matplotlib` | running the test suite and generating plots |
| `all` | numba, sympy, flask, pillow, pytest, matplotlib | everything portable, in one shot |

**CuPy note:** CuPy is tied to your installed CUDA version, so plain `pip install cupy` often isn't what you
want — install the matching wheel by hand instead (e.g. `cupy-cuda12x`). That's why `gpu` is kept out of `all`.

To add a new optional dependency later, add a line to `extras_require` in `setup.py` — nothing else changes,
and the core stays NumPy-only.

## Subsetting & embedding the engine (Pyodide, flat bundles)

If you are carving out a slice of the engine for a constrained target (a Pyodide bundle, a size-limited
deploy), three things matter, and the engine already gives you the tools for each.

**1. The canonical internal import style is packaged path, and it is deliberate.** Every module addresses its
siblings as `from holographic.<family>.holographic_<name> import ...` (2900+ of them; only a dozen legacy
exceptions remain). This is the one blessed style — do not introduce flat `import holographic_<name>` imports,
because the two cannot coexist without a bidirectional shim. If you must run modules *flat* (no `holographic`
package on the path), add a shim that maps the flat names onto the packaged ones once, at bundle-build time,
rather than editing modules:

```python
# flat_shim.py -- run once when building a flat bundle, BEFORE importing any holographic_* module.
import importlib, sys, pkgutil, holographic
for _finder, _name, _ispkg in pkgutil.walk_packages(holographic.__path__, "holographic."):
    if _name.rsplit(".", 1)[-1].startswith("holographic_"):
        sys.modules.setdefault(_name.rsplit(".", 1)[-1], importlib.import_module(_name))
```

**2. Find the TRUE minimal dependency set with `import_footprint`, not a naive tracer.** A follow-every-import
tracer reports ~500 modules for `lecore` because it walks into `try:`-guarded optional-accelerator imports
(numba, cupy, pyfftw, matplotlib, sympy, nltk) that never run on a clean import. The engine's own tracer
separates what actually runs from what is merely referenced:

```python
import lecore
m = lecore.UnifiedMind(dim=64, seed=0)
r = m.import_footprint("lecore")
r["required"]           # ~30 modules that REALLY import (vs r["naive"] ~500)
r["required_external"]  # ['numpy'] -- the real pip dependency for a bundle
r["optional_external"]  # ['PIL', 'cupy', 'numba', 'matplotlib', 'sympy', ...] -- safe to exclude
```

`required_external` is the answer a bundler needs: the core closes over **numpy only**; everything in
`optional_external` is behind a guard and can be dropped from the bundle. (Complements `accelerator_report`,
which says what is *installed here*, and `tools/audit_imports.py`, which says whether an import *resolves*.)

**3. Runtime data files resolve in both layouts already.** Data (the WordNet dictionary, material property
JSON, routing indices) is looked up by trying the `lecore_data` package first, then falling back to a
`__file__`-relative path — so it resolves the same from a clone, a wheel, or a flat bundle without a bare
`import holographic` coupling the lookup to the packaged layout. If you relocate data, keep both lookup arms.

## Where each file goes in the repo

| File | Location |
|---|---|
| `setup.py`, `pyproject.toml`, `lecore.py`, `build_package.sh` | repo root |
| `package.yml` | `.github/workflows/package.yml` |
| `PACKAGING.md` | repo root (or `docs/`) |
