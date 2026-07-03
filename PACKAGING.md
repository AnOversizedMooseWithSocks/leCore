# Packaging leCore as an importable Python package

leCore ships as a normal pip-installable package. After installing it you can:

```python
import lecore
mind = lecore.UnifiedMind(dim=1024, seed=0)      # dim/seed are the first two args; see the class for the rest
# the raw VSA algebra is re-exported for convenience:
c = lecore.bind(lecore.random_vector(1024, 0), lecore.random_vector(1024, 1))
```

The full engine is still importable directly for anything the shim doesn't re-export, e.g.
`from holographic_render import ...` — the modules install at the top level, exactly as they sit in the repo.

## How it's put together (and why)

The engine is ~261 flat `holographic_*.py` modules that import each other by plain names
(`from holographic_ai import bind`). So the package installs them **as top-level modules**, unchanged —
no import rewriting, no `sys.path` tricks. `lecore.py` is a tiny convenience shim that re-exports the main
API so `import lecore` works like any package. Three small files drive it:

- **`setup.py`** — globs `holographic_*.py` at build time into `py_modules` (adding a new module needs no
  edit here), plus the `lecore` shim. Declares `numpy` as the only required dependency; everything else is an
  opt-in extra (see the table below): `jit`, `symbolic`, `gpu`, `ui`, `dev`, and `all`.
- **`pyproject.toml`** — three lines telling `python -m build` to use setuptools.
- **`lecore.py`** — the friendly front door (`UnifiedMind` + the raw ops).

## Building it locally

```sh
pip install build            # one-time
sh build_package.sh          # copies the essentials to a clean folder, strips tests, builds the wheel
pip install dist/lecore-*.whl
```

`build_package.sh` copies **only** the `holographic_*.py` modules (no `test_*`, no app/tour/benchmarks,
no data/docs/figures) into a clean `build_pkg/` folder and builds the wheel from there — so nothing
non-essential leaks into the distribution. (Verified: the wheel contains all 261 modules and zero test files.)

## CI

`package.yml` goes in `.github/workflows/`. On every push to `main` and on version tags (`v*`) it builds the
wheel, **installs it in isolation and imports it as a smoke test**, uploads the wheel + sdist as build
artifacts, and — on a `v*` tag — attaches them to the GitHub Release.

### Versioning
The version lives in `setup.py` (`version="0.1.0"`); bump it per release. If you'd rather have the git tag
drive it, add one line to the workflow before the build step:

```yaml
      - name: Set version from the tag (optional)
        if: startsWith(github.ref, 'refs/tags/v')
        run: sed -i "s/version=\"[^\"]*\"/version=\"${GITHUB_REF_NAME#v}\"/" setup.py
```

## Optional extras (opt-in dependencies)

The core requires **only NumPy**. Everything else is declared as a named "extra" in `setup.py`'s
`extras_require`, so a user opts in with `pip install .[name]` (from the cloned folder) or
`pip install lecore[name]` (if installed from an index). Combine names with commas: `pip install .[ui,jit]`.

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

## Where each file goes in the repo

| File | Location |
|---|---|
| `setup.py`, `pyproject.toml`, `lecore.py`, `build_package.sh` | repo root |
| `package.yml` | `.github/workflows/package.yml` |
| `PACKAGING.md` | repo root (or `docs/`) |
