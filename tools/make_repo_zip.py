#!/usr/bin/env python3
"""make_repo_zip.py -- build the delivery zip, excluding exactly what .gitignore excludes.

WHY THIS EXISTS (a measured leak, kept loud). The delivery zip was built by hand with `zip -qr ... -x
'*__pycache__*' -x '*.pyc'` -- an exclusion list that lived in my head and in a shell line, NOT in the repo.
It shipped, measured: a 7.9 MB `tools/semantic/.knowledge_cache.json` (an embedding cache CI restores from
actions/cache) and five `.gitignore`d BACKLOG docs, in a 27.8 MB archive. Worse, the tree carried a STALE
one-line `.gitignore` (`*.log`) which, on extract, would have OVERWRITTEN the real one -- after which nothing
would have been ignored at all.

So the rule is the same one the doc generators just taught us: A HAND-MAINTAINED SECOND COPY OF A LIST IS
ALWAYS THE STALE ONE. The repo already states what is not source -- that is what `.gitignore` IS. This reads
that file and honours it, so the zip and git agree by construction instead of by my memory.

TWO BUGS THIS FOUND IN .gitignore ITSELF (both fixed there, both worth knowing):
  * `/__pycache__` -- leading slash = ROOT ONLY. Every nested cache was unignored. That is why pycache
    folders kept reappearing. `__pycache__/` matches at any depth.
  * `scripts/.knowledge_cache.json` / `/scripts/nomic_text` -- the semantic tooling MOVED to tools/semantic/
    and the rules did not follow, leaving the big cache unignored.

SUPPORTED SUBSET, honestly stated: comment/blank lines, a leading `/` (anchor to repo root), a trailing `/`
(directory only), `!` negation, and `*`/`?` globs via fnmatch -- which covers every rule this repo's
.gitignore uses. It is NOT a full gitignore engine (no `**` semantics, no per-directory .gitignore files).
If a rule ever needs more, use `git ls-files` instead of teaching this file to be git. KEPT NEGATIVE: `git
ls-files` was the first instinct and is the RIGHT answer in a real checkout -- it is not used here only
because the working tree Claude assembles has no .git metadata, so there is nothing to ask.

    python3 tools/make_repo_zip.py                 # -> repo.zip at the repo root
    python3 tools/make_repo_zip.py --out /tmp/x.zip --dry-run
"""

import argparse
import fnmatch
import os
import sys
import zipfile
from pathlib import Path

# Always excluded regardless of .gitignore: git's own metadata is not source, and the zip must never contain
# a previous zip (that is how an archive doubles in size every round trip).
ALWAYS = [".git/", "repo.zip"]


def repo_root():
    """The repo root -- this script lives in tools/."""
    return Path(__file__).resolve().parent.parent


def load_rules(root):
    """Parse .gitignore into [(pattern, anchored, dir_only, negated)]. Order is preserved: in gitignore the
    LAST matching rule wins, which is what makes `!` negation work."""
    rules = []
    path = root / ".gitignore"
    if not path.exists():
        # Loud, not silent: a missing .gitignore means we have no idea what is junk, so refuse to guess.
        raise SystemExit("no .gitignore at %s -- refusing to guess what is source" % path)
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        anchored = line.startswith("/")
        line = line.lstrip("/")
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if line:
            rules.append((line, anchored, dir_only, negated))
    for extra in ALWAYS:
        rules.append((extra.rstrip("/"), extra.startswith("/"), extra.endswith("/"), False))
    return rules


def _match(rel, is_dir, pattern, anchored, dir_only):
    """Does one rule match this repo-relative path? A non-anchored rule matches at ANY depth (git's rule),
    which is exactly the `/__pycache__` vs `__pycache__/` distinction that leaked caches for months."""
    if dir_only and not is_dir:
        return False
    if anchored:
        # match the whole path, or any parent directory of it, against the pattern
        parts = rel.split("/")
        for i in range(1, len(parts) + 1):
            if fnmatch.fnmatch("/".join(parts[:i]), pattern):
                return True
        return False
    # unanchored: the pattern may match any single component, or a trailing path
    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(os.path.basename(rel), pattern):
        return True
    return any(fnmatch.fnmatch(part, pattern) for part in rel.split("/"))


def ignored(rel, is_dir, rules):
    """True when .gitignore says this path is not source. Last matching rule wins, so `!` can re-include."""
    verdict = False
    for pattern, anchored, dir_only, negated in rules:
        if _match(rel, is_dir, pattern, anchored, dir_only):
            verdict = not negated
    return verdict


def collect(root, rules):
    """Walk the tree and return the sorted list of repo-relative files that BELONG in the zip. Ignored
    directories are pruned, so we never even descend into a 7.9 MB cache directory."""
    keep = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(d for d in dirnames
                             if not ignored((rel_dir + "/" + d).lstrip("/"), True, rules))
        for name in sorted(filenames):
            rel = (rel_dir + "/" + name).lstrip("/")
            if not ignored(rel, False, rules):
                keep.append(rel)
    return sorted(keep)


def build(root=None, out=None, dry_run=False):
    """Write the delivery zip. Returns (path, n_files, n_bytes)."""
    root = Path(root) if root else repo_root()
    out = Path(out) if out else (root / "repo.zip")
    rules = load_rules(root)
    files = collect(root, rules)
    total = sum((root / f).stat().st_size for f in files)
    if dry_run:
        return out, len(files), total
    if out.exists():
        out.unlink()
    # deterministic-ish: sorted entries, stored with the file's own contents; no timestamps are normalised
    # (git does not track mtimes either, and this artifact is a courier, not a gate).
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            z.write(root / rel, rel)
    return out, len(files), total


def main(argv=None):
    """CLI entry: build the zip, or --dry-run to see what would go in."""
    ap = argparse.ArgumentParser(description="Build the delivery zip, honouring .gitignore.")
    ap.add_argument("--out", default=None, help="output path (default: repo.zip at the repo root)")
    ap.add_argument("--dry-run", action="store_true", help="count what would be included, write nothing")
    ap.add_argument("--show-excluded", action="store_true", help="list the biggest things left OUT, and why")
    args = ap.parse_args(argv)

    root = repo_root()
    if args.show_excluded:
        rules = load_rules(root)
        rows = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
            rel_dir = "" if rel_dir == "." else rel_dir
            for d in list(dirnames):
                rel = (rel_dir + "/" + d).lstrip("/")
                if ignored(rel, True, rules):
                    size = sum(f.stat().st_size for f in (root / rel).rglob("*") if f.is_file())
                    rows.append((size, rel + "/"))
                    dirnames.remove(d)
            for name in filenames:
                rel = (rel_dir + "/" + name).lstrip("/")
                if ignored(rel, False, rules):
                    rows.append(((root / rel).stat().st_size, rel))
        for size, rel in sorted(rows, reverse=True)[:25]:
            print("  %9.1f KB  %s" % (size / 1024.0, rel))
        print("  -- %d path(s) excluded, %.1f MB total" % (len(rows), sum(r[0] for r in rows) / 1e6))
        return 0

    out, n, total = build(out=args.out, dry_run=args.dry_run)
    verb = "would include" if args.dry_run else "wrote %s --" % out
    print("%s %d files, %.1f MB uncompressed" % (verb, n, total / 1e6))
    if not args.dry_run:
        print("  archive: %.1f MB" % (out.stat().st_size / 1e6))
    return 0


def _selftest():
    """Assert the REAL contract on the live tree: the known junk is OUT and the source is IN."""
    root = repo_root()
    rules = load_rules(root)

    # the two leaks that actually shipped, pinned by name
    assert ignored("tools/semantic/.knowledge_cache.json", False, rules), "the 7.9 MB cache must be excluded"
    assert ignored("docs/BACKLOG_modeling.md", False, rules), "backlog docs must be excluded"
    # the nested-pycache bug: a leading-slash rule would let these through
    for d in ("holographic/__pycache__", "tests/__pycache__", "benchmarks/__pycache__"):
        assert ignored(d, True, rules), "%s must be excluded at ANY depth, not just the root" % d
    assert ignored("holographic/misc/__pycache__/x.pyc", False, rules)
    assert ignored("repo.zip", False, rules) and ignored(".git", True, rules)

    # ...and the source must survive: excluding real code would be a far worse bug than shipping a cache
    for keep in ("holographic/misc/holographic_unified.py", "tools/regen_docs.py", "README.md",
                 "lecore_data/routing/index_128d.npz", ".github/workflows/ci.yml", "tests/test_regen_docs.py"):
        assert not ignored(keep, False, rules), "%s is SOURCE and must be kept" % keep

    files = collect(root, rules)
    assert any(f.endswith("holographic_unified.py") for f in files), "the engine must be in the zip"
    assert not any("__pycache__" in f or f.endswith(".pyc") for f in files), "cache leaked into the file list"
    assert not any("knowledge_cache" in f for f in files), "the embedding cache leaked into the file list"
    print("OK: make_repo_zip self-test passed (%d files would ship; junk excluded, source intact)" % len(files))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
