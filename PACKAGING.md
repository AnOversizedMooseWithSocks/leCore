# Packaging leCore as an importable Python package

leCore ships as a normal pip-installable package. On PyPI the project is published as **`leos-core`** (the
plain name `lecore` was already taken by an unrelated project, and this engine is the core of the larger
**leOS** project). The important part: the *distribution* name and the *import* name are independent, so you
install one name and import another:

```sh
pip install leos-core        # the distribution name (what PyPI knows it as)
```
```python
import lecore                 # the import name is unchanged -- the modules install at the top level
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
  opt-in extra (see the table below): `jit`, `symbolic`, `gpu`, `ui`, `dev`, and `all`. It also ships the
  **`lecore_data`** package (below) via `package_data`, so the runtime data travels with the wheel.
- **`pyproject.toml`** — three lines telling `python -m build` to use setuptools.
- **`lecore.py`** — the friendly front door (`UnifiedMind` + the raw ops).
- **`lecore_data/`** — a small importable package holding the data the engine reads **at runtime** (the WordNet
  dictionary the `lookup` faculty uses; the material-property JSON the heat model uses). It is a *package* rather
  than a loose `data/` folder for one concrete reason: with the flat `py_modules` layout, a bare `data/` directory
  is **not** carried into a wheel, but a package is. `import lecore_data` resolves the same from a clone and an
  install, so the loaders (`holographic_dictionary`, `holographic_heat`) find their files either way. Demo-only
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

## Where each file goes in the repo

| File | Location |
|---|---|
| `setup.py`, `pyproject.toml`, `lecore.py`, `build_package.sh` | repo root |
| `package.yml` | `.github/workflows/package.yml` |
| `PACKAGING.md` | repo root (or `docs/`) |
