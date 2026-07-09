# leCore Development Strategy — build without gaps, wire holographically

This is the standard process for changing leCore. It exists because the failure mode of this codebase is not
bugs — it's **gaps**: capabilities that get built but never wired to a faculty, examples that rot, modules that
become undiscoverable, and code that only ever runs inside a test. The rule of thumb: *a feature that
`find_capability` can't surface, and `/invoke` can't call, does not exist.*

Follow these steps for every change. leCore is a participant in the process, not just the subject of it.

---

## 0. Audit first — ask leCore what already exists

Before writing anything, query the running system for the capability you think you need:

```python
import lecore
mind = lecore.UnifiedMind(dim=512, seed=0)
for q in ["edit a source file", "run a long job in the background", "compare two renders"]:
    print(q, "->", mind.find_capability(q)[:3])
```

Most "new" work is already present under a different name (this session: job control, image compare, and the
message bus all already existed — only file-editing was a genuine gap). If `find_capability` returns something
relevant, **extend or reuse it**. Only build new when the audit shows the top hits are unrelated fallbacks.

Reach for the agentic tools to do this exploration — they're the same ones an external agent uses, so dogfooding
them keeps them honest:

```python
mind.set_file_root(".")
mind.file_grep("def cloud_field")          # where does it live?
mind.file_view("path/to/mod.py", 40, 80)   # look before editing (line-numbered)
```

---

## 1. Build the capability in its module, with a `_selftest()`

Put the logic in the right family package (`holographic/<family>/holographic_*.py`). Every module ends with a
`def _selftest()` that asserts the real behaviour and a `if __name__ == "__main__": _selftest()`. The self-test is
not optional — it's what lets the reachability audit and CI trust the module. Write it to **fail loudly** on the
thing most likely to break (this session, the render-job self-test caught a real defaults-mismatch bug, and the
noise self-test now guards the fast-bake's exactness to 1e-10).

Give every public function a real docstring. A module with no docstring is **undiscoverable** by
`find_capability` — the audit flags this as a hard error.

---

## 2. Wire it to a faculty — never leave it import-only

A module reachable only by `import` is a gap. Expose it through a method on `UnifiedMind`
(`holographic/misc/holographic_unified.py`), delegating to the module:

```python
def file_replace(self, path, old, new, count=1):
    """<one-line what + when to use>. See holographic_codeedit.Editor.replace."""
    return self._editor.replace(path, old, new, count=count)
```

This is load-bearing: the HTTP service auto-introspects every public mind method into `GET /tools`, so **any mind
method is instantly agent-invokable via `POST /invoke`** with no extra work. A capability that isn't a mind method
can't be called by an agent over the wire.

---

## 3. Register it in the catalog so it's discoverable

Add a `register_capability(...)` entry in `holographic/caching_and_storage/holographic_catalog.py` with a
plain-language description, a **runnable** `example=`, and generous `aliases=` (the phrases a user or agent would
actually search). This is what makes step 0 work for the *next* person. Verify:

```python
mind.find_capability("the phrasing a user would use")   # must return your capability, not a fallback
```

---

## 4. Verify — statically, in-process, and end-to-end

Run all three, in order. Static alone is not enough (it has repeatedly passed code that was functionally broken):

```bash
# static: syntax / compile / broken-import across the whole tree
python3 -c "from reorganize_repo import analyze_repo, verify, DEFAULT_IGNORE; from pathlib import Path; \
f=analyze_repo(Path('src'), DEFAULT_IGNORE); r=verify(Path('src'), f); \
print(len(f), len(r['syntax_errors']), len(r['compile_errors']), len(r['broken_local_imports']))"

# module self-test
python3 -m holographic.<family>.holographic_<name>      # runs _selftest()

# end-to-end through the mind (and, for agent-facing work, over HTTP /invoke)
python3 -c "import lecore; m=lecore.UnifiedMind(dim=256, seed=0); ...your call..."
```

For anything agent-facing, prove the **full HTTP round-trip** once: start the service, `POST /invoke` your method,
confirm the result. "It works in-process" and "an agent can call it" are different claims.

While editing, use `mind.file_python_check(path)` right after each edit — catch a broken edit immediately instead
of at import time.

---

## 5. Run the wiring audits — close the gaps you just might have opened

```bash
python3 tools/reachability_audit.py     # is anything unwired / undocumented / buried?
python3 tools/catalog_gaps.py           # capability without a home / example?
python3 tools/skill_lint.py             # every catalog example still resolves + runs?
```

Target state: 0 missing docstrings, 0 catalog gaps, 0 invocation gaps. If your new module shows up under
"IMPORT-ONLY, not a declared negative", go back to step 2 — it isn't wired. If it's a *deliberate* non-faculty,
record it as a declared negative so the audit stays meaningful.

---

## 6. Keep the generated docs in sync

If you touched the catalog, regenerate and let the CI drift-gate keep everyone honest:

```bash
python3 capdoc.py     # rewrites CAPABILITIES.md + capabilities.json from the live catalog
python3 docgen.py     # rewrites REFERENCE.md from module docstrings
```

`capabilities.json` is the machine-readable contract other tools ingest; the CI gate fails if it drifts from the
catalog, so a capability change that isn't regenerated can't merge. Add runnable, verified snippets to the
relevant guide (`RENDERING_GUIDE.md`, `FEATURE_GUIDE.md`) — a doc example that isn't run is a doc example that
will rot.

---

## The checklist (paste into the PR)

- [ ] **Audited** with `find_capability` before building; reused what existed.
- [ ] Logic in the right family module, with a `_selftest()` that fails loudly, and docstrings on all public defs.
- [ ] **Wired** to a `UnifiedMind` method (so it's `/invoke`-able) — nothing left import-only.
- [ ] **Registered** in the catalog with a runnable example + search aliases; confirmed discoverable.
- [ ] Verified **static + self-test + end-to-end** (and HTTP `/invoke` if agent-facing); `file_python_check` clean.
- [ ] `reachability_audit` / `catalog_gaps` / `skill_lint` all clean.
- [ ] Generated docs regenerated (`capdoc.py`, `docgen.py`); guide snippets are runnable and were run.

If every box is checked, the capability is real: discoverable, callable, wired, documented, and un-rottable — not
isolated in a test.
