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
  opt-in extra (see the table below): `jit`, `fft`, `symbolic`, `zig`, `gpu`, `ui`, `images`, `dev`, and `all`. It also ships the
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

## CI: auto-publish on every green merge to main

Releases are hands-off. There are **no tags to manage** and **one number to edit**.

**The version lives in one file: `VERSION`** (currently `0.2.0`). Everything reads it — `setup.py` (`version=read_version()`),
`lecore.__version__` (from the installed wheel's metadata, or `VERSION` from a clone), and `tools/check_version.py`.
They can never drift again (previously `lecore.py` was stuck at `0.1.0` while `setup.py` said `0.2.0`).

**You own `major.minor`; CI owns `patch`.** To cut a `0.3` or `1.0` line, hand-edit `VERSION` (e.g. to `0.3.0`) and
commit — the next merge auto-bumps from there (`0.3.1`, `0.3.2`, …). The automation only ever touches the third digit.

The flow, end to end:

1. **Branch protection makes tests a merge gate.** A branch can only merge to `main` once the `tests` check passes.
   This is a one-time repo setting (below), enforced by GitHub, not by a workflow file.
2. **A merge lands on `main`** → the `tests` workflow runs on `main` and (if green) completes successfully.
3. **`package.yml` triggers on that completion** (`workflow_run`), and only if the run it followed **succeeded** and
   was **on main**. It runs `tools/bump_version.py` (patch bump, `0.2.0` → `0.2.1`), commits the new `VERSION` back to
   `main` with `[skip ci]` in the message (so the commit does **not** re-trigger tests → no publish loop), builds and
   smoke-tests the wheel, and publishes it to PyPI.

So the moment a green PR merges, `pip install -U leos-core` gets a fresh patch release. A failed test run on main
publishes nothing (the `conclusion == 'success'` gate). A push to a feature branch publishes nothing (the
`head_branch == 'main'` gate).

> **Manual re-publish:** the workflow also has a `workflow_dispatch` button (Actions tab → package → Run workflow).
> That path does **not** bump — it re-publishes the current `VERSION` as-is, for the rare case a publish failed *after*
> the bump commit already landed.

### One-time setup

**(a) Branch protection** — Settings → Branches → add a rule for `main`: require status checks to pass before
merging, and select the **`pytest`** check (the `tests` workflow's per-change job). Optionally require a PR before
merging. That is what makes "can only merge when tests pass" real.

**(b) PyPI Trusted Publishing** (OpenID Connect — no API token or password stored in the repo; that's what the
workflow's `id-token: write` permission is for). One-time on PyPI:

1. Log in at <https://pypi.org>. (For a dry run first, do the same on <https://test.pypi.org> and uncomment the
   `repository-url` line in the workflow.)
2. Because `leos-core` doesn't exist on PyPI yet, add a **pending publisher**: Account menu → **Publishing** → "Add a
   new pending publisher". (After the first successful publish it becomes a normal publisher on the project.)
3. Fill in exactly:
   - **PyPI Project Name:** `leos-core`
   - **Owner:** `AnOversizedMooseWithSocks`
   - **Repository name:** `leCore`
   - **Workflow name:** `package.yml`
   - **Environment name:** leave blank.
4. Save. From then on, every green merge to `main` publishes the next patch automatically.

Until step (b) is done, the publish job will *run* on a green merge but *fail* at the PyPI handshake — the bump and
commit still happen, so just complete the PyPI setup and the next merge (or a manual dispatch) publishes.

### Versioning helpers

`python tools/check_version.py` prints the current version (reads `VERSION`). `python tools/bump_version.py --print`
shows what the next patch would be without writing; `--current` prints the current; no flag performs the bump. The
bump tool refuses a malformed `VERSION` (exit 1) rather than publishing a garbage number.

## Optional extras (opt-in dependencies)

The core requires **only NumPy**. Everything else is declared as a named "extra" in `setup.py`'s
`extras_require`, so a user opts in with `pip install .[name]` (from the cloned folder) or
`pip install leos-core[name]` (if installed from PyPI). Combine names with commas: `pip install .[ui,jit]`.

| Extra | Pulls in | For |
|---|---|---|
| `jit` | `numba` | numba-compiled fast paths (`holographic_jit`, `sdf_render`, `codegen`) |
| `fft` | `pyfftw` | FFTW-backed FFT with plan caching (`holographic_fft`), opt-in via `mind.fft_backend(use_pyfftw=True)`; NumPy FFT stays the deterministic default |
| `symbolic` | `sympy` | design-time symbolic gradients (`holographic_codegen`, `sdf_render`) |
| `zig` | `ziglang` | native batch kernels + raymarcher (`holographic_zigrun`, `zigmarch`); ships the whole Zig toolchain, no system compiler needed |
| `gpu` | `cupy` | the GPU backend (`holographic_backend`) — see the CuPy note |
| `ui` | `flask`, `pillow` | the browser UI (`app.py`) and image load/save |
| `images` | `pillow` | image I/O beyond stdlib PNG (jpg/webp/…) without pulling in Flask — a headless subset of `ui` |
| `dev` | `pytest`, `matplotlib`, `nltk` | running the test suite, generating plots, and loading the text corpora the benchmarks/ablations use |
| `all` | numba, pyfftw, sympy, flask, pillow, pytest, matplotlib, ziglang, nltk | everything portable, in one shot (CuPy excluded) |

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
