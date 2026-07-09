"""holographic_codeedit.py -- structured FILE / CODE editing for an agent working on a codebase (this is the tool
leCore was missing: everything else could read assets or run allowlisted programs, but nothing could safely
read/write/patch a source file). Read, write, exact-string replace, line insert/delete, delete, archive, grep, and
list -- each scoped to a ROOT directory so an agent can't wander outside the project.

DESIGN
------
  * ROOT SANDBOX. Every path is resolved and must stay inside `root` (default: the current working directory). A
    path that escapes via .. or an absolute path elsewhere raises EditError -- the single safety gate, the same
    "there is no path from a caller to somewhere it shouldn't reach" spirit as holographic_command's allowlist.
  * EXACT, REVIEWABLE EDITS. `replace` requires its old text to occur EXACTLY ONCE (like a good patch tool) so an
    edit can never silently hit the wrong place; it returns the 1-based line where it landed. Every mutating call
    returns a small dict describing what changed, so an agent (or an /invoke caller) can verify the effect.
  * ATOMIC WRITES. A write goes to a temp file in the same directory then os.replace()s into place, so a crash
    mid-write can't leave a half-written source file.
  * NON-DESTRUCTIVE DELETE. `archive` moves a file into `.lecore_archive/<timestamp>/` (preserving its relative
    path) instead of destroying it -- an "undo" for the agent. `delete` really removes, and is separate on purpose.

Wire the Editor's methods onto UnifiedMind (see file_read / file_write / file_replace / ... there) and they become
callable over the HTTP tool protocol (GET /tools, POST /invoke) like any other faculty -- the point of this module
is to make an agent's normal file work a first-class, introspectable part of the same fabric.
"""
import os
import shutil
import time


class EditError(Exception):
    """A file edit could not be performed safely (path escaped the root, target not found, ambiguous replace, ...)."""


class Editor:
    """Root-scoped file operations. Construct with the project root; every method takes paths RELATIVE to it."""

    def __init__(self, root="."):
        self.root = os.path.abspath(root)

    # -- the safety gate ------------------------------------------------------------------------------------
    def _resolve(self, relpath):
        """Resolve `relpath` against the root and REFUSE anything that escapes it. Returns the absolute path."""
        full = os.path.abspath(os.path.join(self.root, relpath))
        # os.path.commonpath raises on mixed drives; guard with a prefix check on the normalised paths
        if full != self.root and not full.startswith(self.root + os.sep):
            raise EditError("path %r escapes the editor root %r" % (relpath, self.root))
        return full

    def _rel(self, full):
        return os.path.relpath(full, self.root)

    # -- read / inspect -------------------------------------------------------------------------------------
    def read(self, relpath, max_bytes=1_000_000):
        """Return a file's text (utf-8). Raises EditError if it's missing or larger than max_bytes."""
        full = self._resolve(relpath)
        if not os.path.isfile(full):
            raise EditError("no such file: %r" % relpath)
        size = os.path.getsize(full)
        if size > max_bytes:
            raise EditError("file %r is %d bytes (> max_bytes=%d); read a slice or raise the limit"
                            % (relpath, size, max_bytes))
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def read_lines(self, relpath, start=1, end=None):
        """Return lines [start, end] (1-based, inclusive) of a file as a list of strings (newlines stripped).
        `end=None` reads to the end. Handy for an agent to look at just a region before editing it."""
        text = self.read(relpath)
        lines = text.splitlines()
        s = max(1, int(start)) - 1
        e = len(lines) if end is None else min(len(lines), int(end))
        return lines[s:e]

    def view(self, relpath, start=1, end=None):
        """Return lines [start, end] (1-based inclusive) as a single string WITH LINE NUMBERS prefixed, exactly the
        form an agent needs to then target replace/insert/delete_lines/replace_lines. `end=None` -> to EOF. This is
        the everyday 'show me this region so I can edit it' call (read() gives raw text; view() gives located text)."""
        text = self.read(relpath)
        lines = text.splitlines()
        s = max(1, int(start))
        e = len(lines) if end is None else min(len(lines), int(end))
        width = len(str(e))
        return "\n".join("%*d\t%s" % (width, i, lines[i - 1]) for i in range(s, e + 1))

    def read_many(self, relpaths, max_bytes=1_000_000):
        """Read several files at once -> {relpath: text}. Saves an agent a round-trip per file when gathering
        context. A file that can't be read maps to an "<error: ...>" string rather than aborting the whole call."""
        out = {}
        for p in relpaths:
            try:
                out[p] = self.read(p, max_bytes=max_bytes)
            except EditError as e:
                out[p] = "<error: %s>" % e
        return out

    def count_occurrences(self, relpath, text):
        """How many times `text` occurs in a file -- check BEFORE a replace to know whether it's unique (count==1),
        absent (0), or needs count=N/0. Cheap way to avoid an ambiguous-replace EditError."""
        return self.read(relpath).count(text)

    def exists(self, relpath):
        """True if the (root-scoped) path exists."""
        try:
            return os.path.exists(self._resolve(relpath))
        except EditError:
            return False

    def list_dir(self, relpath=".", recursive=False, suffix=None):
        """List files under a directory (relative paths), optionally recursively and filtered by suffix (e.g.
        '.py'). Skips __pycache__ and hidden dirs so an agent sees source, not noise."""
        base = self._resolve(relpath)
        if not os.path.isdir(base):
            raise EditError("not a directory: %r" % relpath)
        out = []
        if recursive:
            for dp, dns, fns in os.walk(base):
                dns[:] = [d for d in dns if d != "__pycache__" and not d.startswith(".")]
                for fn in fns:
                    if suffix is None or fn.endswith(suffix):
                        out.append(self._rel(os.path.join(dp, fn)))
        else:
            for name in sorted(os.listdir(base)):
                if suffix is None or name.endswith(suffix):
                    out.append(self._rel(os.path.join(base, name)))
        return sorted(out)

    def grep(self, pattern, relpath=".", suffix=".py", max_hits=200):
        """Plain-substring search across files under `relpath` (filtered by `suffix`). Returns a list of
        {file, line, text} for each match -- the 'find where X is used' an agent reaches for constantly."""
        base = self._resolve(relpath)
        hits = []
        walk_root = base if os.path.isdir(base) else os.path.dirname(base)
        for dp, dns, fns in os.walk(walk_root):
            dns[:] = [d for d in dns if d != "__pycache__" and not d.startswith(".")]
            for fn in fns:
                if suffix and not fn.endswith(suffix):
                    continue
                full = os.path.join(dp, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                hits.append({"file": self._rel(full), "line": i, "text": line.rstrip("\n")[:300]})
                                if len(hits) >= max_hits:
                                    return hits
                except OSError:
                    continue
        return hits

    # -- write / create -------------------------------------------------------------------------------------
    def _atomic_write(self, full, text):
        # snapshot the PRIOR contents (if any) onto the undo stack before overwriting, so any mutating op is
        # reversible with undo(). Cap the stack so a long session can't grow without bound.
        if not hasattr(self, "_undo"):
            self._undo = []
        prior = None
        if os.path.isfile(full):
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    prior = f.read()
            except OSError:
                prior = None
        self._undo.append((full, prior))
        if len(self._undo) > 100:
            self._undo.pop(0)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        tmp = full + ".lecore_tmp_%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, full)

    def undo(self, steps=1):
        """Reverse the last `steps` mutating file operations (write/replace/insert/delete_lines/replace_lines),
        restoring each file to its prior contents -- a real undo for the agent. A file that didn't exist before
        the edit is removed again. Returns {undone, files}."""
        if not hasattr(self, "_undo"):
            self._undo = []
        done = []
        for _ in range(int(steps)):
            if not self._undo:
                break
            full, prior = self._undo.pop()
            if prior is None:
                if os.path.isfile(full):
                    os.remove(full)
            else:
                tmp = full + ".lecore_undo_%d" % os.getpid()
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(prior)
                os.replace(tmp, full)
            done.append(self._rel(full))
        return {"undone": len(done), "files": done}

    def write(self, relpath, text, overwrite=True):
        """Create or replace a file with `text` (atomically). With overwrite=False, refuses to clobber an existing
        file. Returns {path, bytes, created}."""
        full = self._resolve(relpath)
        existed = os.path.exists(full)
        if existed and not overwrite:
            raise EditError("file %r exists and overwrite=False" % relpath)
        self._atomic_write(full, text)
        return {"path": relpath, "bytes": len(text.encode("utf-8")), "created": not existed}

    def replace(self, relpath, old, new, count=1):
        """Replace EXACT text `old` with `new` in a file. By default `old` must occur EXACTLY ONCE (count=1) so the
        edit is unambiguous; pass count=0 to replace ALL occurrences, or count=N to require exactly N. Returns
        {path, replacements, first_line}. This is the workhorse edit -- the same contract as a careful patch tool."""
        text = self.read(relpath)
        n = text.count(old)
        if n == 0:
            raise EditError("old text not found in %r" % relpath)
        if count and n != count:
            raise EditError("old text occurs %d times in %r but count=%d was required (make it unique or set count)"
                            % (n, relpath, count))
        replaced = text.replace(old, new) if count == 0 else text.replace(old, new, count)
        # 1-based line of the first occurrence, for the agent to jump to
        first_line = text[:text.find(old)].count("\n") + 1
        self._atomic_write(self._resolve(relpath), replaced)
        return {"path": relpath, "replacements": (n if count == 0 else count), "first_line": first_line}

    def insert(self, relpath, after_line, text):
        """Insert `text` (one or more lines) AFTER 1-based line `after_line` (0 = at the very top). Returns
        {path, inserted_at}. Newlines in `text` are honoured; a trailing newline is added if missing."""
        lines = self.read(relpath).splitlines(keepends=True)
        idx = max(0, min(int(after_line), len(lines)))
        block = text if text.endswith("\n") else text + "\n"
        lines[idx:idx] = [block]
        self._atomic_write(self._resolve(relpath), "".join(lines))
        return {"path": relpath, "inserted_at": idx + 1}

    def replace_lines(self, relpath, start, end, text):
        """Replace lines [start, end] (1-based inclusive) with `text` -- the range-based edit to reach for when the
        old content ISN'T unique enough for replace() (e.g. a body of boilerplate). Pair it with view() to get the
        line numbers first. Returns {path, replaced, new_lines}."""
        lines = self.read(relpath).splitlines(keepends=True)
        s = max(1, int(start)) - 1
        e = min(len(lines), int(end))
        if s >= e:
            raise EditError("empty or inverted line range [%s, %s] in %r" % (start, end, relpath))
        block = text if text.endswith("\n") else text + "\n"
        removed = e - s
        lines[s:e] = [block]
        self._atomic_write(self._resolve(relpath), "".join(lines))
        return {"path": relpath, "replaced": removed, "new_lines": block.count("\n")}

    def delete_lines(self, relpath, start, end):
        """Delete lines [start, end] (1-based, inclusive). Returns {path, deleted}."""
        lines = self.read(relpath).splitlines(keepends=True)
        s = max(1, int(start)) - 1
        e = min(len(lines), int(end))
        if s >= e:
            raise EditError("empty or inverted line range [%s, %s] in %r" % (start, end, relpath))
        deleted = e - s
        del lines[s:e]
        self._atomic_write(self._resolve(relpath), "".join(lines))
        return {"path": relpath, "deleted": deleted}

    # -- delete / archive -----------------------------------------------------------------------------------
    def archive(self, relpath, archive_dir=".lecore_archive"):
        """Move a file into `archive_dir/<timestamp>/<its relative path>` instead of deleting it -- a reversible
        'delete' for an agent. Returns {archived_from, archived_to}."""
        full = self._resolve(relpath)
        if not os.path.isfile(full):
            raise EditError("no such file: %r" % relpath)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = self._resolve(os.path.join(archive_dir, stamp, relpath))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(full, dest)
        return {"archived_from": relpath, "archived_to": self._rel(dest)}

    def delete(self, relpath):
        """Permanently remove a file (use archive() if you might want it back). Returns {deleted}."""
        full = self._resolve(relpath)
        if not os.path.isfile(full):
            raise EditError("no such file: %r" % relpath)
        os.remove(full)
        return {"deleted": relpath}

    def move(self, src, dst, overwrite=False):
        """Move/rename a file within the root. Returns {moved_from, moved_to}."""
        s = self._resolve(src); d = self._resolve(dst)
        if not os.path.exists(s):
            raise EditError("no such file: %r" % src)
        if os.path.exists(d) and not overwrite:
            raise EditError("destination %r exists and overwrite=False" % dst)
        os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
        shutil.move(s, d)
        return {"moved_from": src, "moved_to": dst}

    def find_definition(self, name, relpath=".", suffix=".py"):
        """Find where a Python function or class `name` is DEFINED under `relpath` (matches `def name`/`class name`),
        returning [{file, line, kind, text}]. The 'jump to definition' an agent needs to stop grepping blindly."""
        hits = []
        base = self._resolve(relpath)
        walk_root = base if os.path.isdir(base) else os.path.dirname(base)
        needles = (("def " + name + "(", "function"), ("def " + name + " (", "function"),
                   ("class " + name + "(", "class"), ("class " + name + ":", "class"),
                   ("class " + name + " ", "class"))
        for dp, dns, fns in os.walk(walk_root):
            dns[:] = [d for d in dns if d != "__pycache__" and not d.startswith(".")]
            for fn in fns:
                if suffix and not fn.endswith(suffix):
                    continue
                full = os.path.join(dp, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            ls = line.lstrip()
                            for needle, kind in needles:
                                if ls.startswith(needle):
                                    hits.append({"file": self._rel(full), "line": i, "kind": kind,
                                                 "text": line.rstrip("\n")[:200]})
                                    break
                except OSError:
                    continue
        return hits

    def replace_across(self, old, new, relpath=".", suffix=".py", dry_run=False):
        """Replace EXACT text `old` with `new` in EVERY file under `relpath` that contains it (all occurrences per
        file). Returns [{file, replacements}]. With dry_run=True, reports what WOULD change without writing -- the
        safe way to preview a codebase-wide rename before committing. This is one undo step PER file changed."""
        base = self._resolve(relpath)
        walk_root = base if os.path.isdir(base) else os.path.dirname(base)
        results = []
        for dp, dns, fns in os.walk(walk_root):
            dns[:] = [d for d in dns if d != "__pycache__" and not d.startswith(".")]
            for fn in fns:
                if suffix and not fn.endswith(suffix):
                    continue
                full = os.path.join(dp, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except OSError:
                    continue
                n = text.count(old)
                if n == 0:
                    continue
                results.append({"file": self._rel(full), "replacements": n})
                if not dry_run:
                    self._atomic_write(full, text.replace(old, new))
        return results

    def tree(self, relpath=".", max_depth=3, suffix=None):
        """An indented directory TREE under `relpath` (skips __pycache__/hidden), to `max_depth` levels -- the
        'show me the layout' an agent wants before navigating. Returns a single string."""
        base = self._resolve(relpath)
        if not os.path.isdir(base):
            raise EditError("not a directory: %r" % relpath)
        lines = []

        def walk(d, depth, prefix):
            if depth > max_depth:
                return
            try:
                entries = sorted(os.listdir(d))
            except OSError:
                return
            dirs = [e for e in entries if os.path.isdir(os.path.join(d, e))
                    and e != "__pycache__" and not e.startswith(".")]
            files = [e for e in entries if os.path.isfile(os.path.join(d, e))
                     and (suffix is None or e.endswith(suffix))]
            for e in dirs:
                lines.append("%s%s/" % (prefix, e))
                walk(os.path.join(d, e), depth + 1, prefix + "  ")
            for e in files:
                lines.append("%s%s" % (prefix, e))

        lines.append(self._rel(base) + "/" if base != self.root else "./")
        walk(base, 1, "  ")
        return "\n".join(lines)

    def import_check(self, relpath):
        """Deeper than python_check: actually IMPORT the module (as a dotted path under the root) in a subprocess
        and report success or the real ImportError/traceback tail. Catches broken imports and load-time errors that
        a syntax check misses. Returns {ok, error}. Runs in a fresh process so it can't pollute the caller."""
        import subprocess, sys
        full = self._resolve(relpath)
        if not full.endswith(".py"):
            raise EditError("import_check is for .py files: %r" % relpath)
        rel = os.path.relpath(full, self.root)
        dotted = rel[:-3].replace(os.sep, ".")
        code = ("import sys; sys.path.insert(0, %r)\n"
                "import importlib\n"
                "try:\n"
                "    importlib.import_module(%r)\n"
                "    print('OK')\n"
                "except Exception as e:\n"
                "    import traceback; traceback.print_exc()\n" % (self.root, dotted))
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "import timed out (120s)"}
        if r.returncode == 0 and r.stdout.strip().endswith("OK"):
            return {"ok": True, "error": None}
        tail = (r.stderr.strip() or r.stdout.strip()).splitlines()
        return {"ok": False, "error": "\n".join(tail[-4:]) if tail else "unknown import failure"}

    def python_check(self, relpath):
        """Parse a .py file with the ast module and report whether it's syntactically valid -- the check to run
        RIGHT AFTER editing a Python file, so a broken edit is caught immediately instead of at import time.
        Returns {ok: bool, error: None | "line L: message"}. (Syntax only; it does not import or execute.)"""
        import ast
        src = self.read(relpath)
        try:
            ast.parse(src)
            return {"ok": True, "error": None}
        except SyntaxError as e:
            return {"ok": False, "error": "line %s: %s" % (e.lineno, e.msg)}


def _selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="lecore_codeedit_")
    ed = Editor(root)

    # write + read
    ed.write("pkg/mod.py", "def a():\n    return 1\n\ndef b():\n    return 2\n")
    assert "def a()" in ed.read("pkg/mod.py")

    # the safety gate: escaping the root is refused
    for bad in ("../outside.py", "/etc/passwd"):
        try:
            ed.read(bad); assert False, "escape %r should have raised" % bad
        except EditError:
            pass

    # exact replace: unique required
    r = ed.replace("pkg/mod.py", "return 1", "return 42")
    assert r["replacements"] == 1 and ed.read("pkg/mod.py").count("return 42") == 1
    assert r["first_line"] == 2
    # ambiguous replace refused
    ed.write("dup.py", "x\nx\n")
    try:
        ed.replace("dup.py", "x", "y"); assert False
    except EditError:
        pass
    assert ed.replace("dup.py", "x", "y", count=0)["replacements"] == 2   # replace-all is allowed explicitly

    # insert + delete_lines + read_lines
    ed.write("lines.txt", "a\nb\nc\n")
    ed.insert("lines.txt", 1, "INSERTED")
    assert ed.read_lines("lines.txt", 1, 2) == ["a", "INSERTED"]
    ed.delete_lines("lines.txt", 2, 2)
    assert "INSERTED" not in ed.read("lines.txt")

    # view (line-numbered), read_many, count_occurrences, replace_lines, python_check -- the ergonomics additions
    v = ed.view("pkg/mod.py", 1, 2)
    assert "\t" in v and v.splitlines()[0].strip().startswith("1")     # numbered
    many = ed.read_many(["pkg/mod.py", "nope.py"])
    assert "def a" in many["pkg/mod.py"] and many["nope.py"].startswith("<error")
    assert ed.count_occurrences("pkg/mod.py", "def ") == 2
    ed.write("range.py", "L1\nL2\nL3\nL4\n")
    rr = ed.replace_lines("range.py", 2, 3, "NEW2\nNEW3")
    assert rr["replaced"] == 2 and ed.read_lines("range.py", 2, 3) == ["NEW2", "NEW3"]
    assert ed.python_check("pkg/mod.py")["ok"] is True
    ed.write("broken.py", "def f(:\n")
    chk = ed.python_check("broken.py")
    assert chk["ok"] is False and "line" in chk["error"]

    # undo: an edit is reversible; a freshly-created file is removed on undo
    ed.write("undo_me.py", "original\n")
    ed.replace("undo_me.py", "original", "changed")
    assert "changed" in ed.read("undo_me.py")
    ed.undo()
    assert ed.read("undo_me.py") == "original\n"          # the replace was reversed
    ed.write("brand_new.py", "x\n")
    ed.undo()
    assert not ed.exists("brand_new.py")                   # undo of a create removes the file

    # find_definition
    ed.write("defs.py", "import os\n\ndef alpha():\n    return 1\n\nclass Beta:\n    pass\n")
    fd = ed.find_definition("alpha")
    assert any(h["file"] == "defs.py" and h["kind"] == "function" for h in fd)
    assert any(h["kind"] == "class" for h in ed.find_definition("Beta"))

    # replace_across: dry-run reports, real run edits every file
    ed.write("a1.py", "call_old()\n"); ed.write("sub/a2.py", "x = call_old() + call_old()\n")
    preview = ed.replace_across("call_old", "call_new", dry_run=True)
    assert sum(r["replacements"] for r in preview) == 3 and ed.read("a1.py") == "call_old()\n"  # not yet changed
    done = ed.replace_across("call_old", "call_new")
    assert "call_new" in ed.read("a1.py") and "call_new" in ed.read("sub/a2.py")

    # tree + import_check
    t = ed.tree(".")
    assert "pkg/" in t and "defs.py" in t
    ed.write("good_mod.py", "VALUE = 42\n")
    assert ed.import_check("good_mod.py")["ok"] is True
    ed.write("bad_import.py", "import a_module_that_does_not_exist_xyz\n")
    assert ed.import_check("bad_import.py")["ok"] is False

    # grep + list
    hits = ed.grep("def b", suffix=".py")
    assert any(h["file"] == "pkg/mod.py" for h in hits)
    assert "pkg/mod.py" in ed.list_dir(".", recursive=True, suffix=".py")

    # archive is reversible-friendly (file leaves its spot, lands under the archive dir), delete is final
    a = ed.archive("dup.py")
    assert not ed.exists("dup.py") and ed.exists(a["archived_to"])
    ed.write("gone.py", "temp\n"); ed.delete("gone.py")
    assert not ed.exists("gone.py")

    # move
    ed.write("old_name.py", "keep\n"); ed.move("old_name.py", "sub/new_name.py")
    assert not ed.exists("old_name.py") and ed.read("sub/new_name.py") == "keep\n"

    print("OK: holographic_codeedit self-test passed (root sandbox blocks escapes; unique/all replace; insert/"
          "delete_lines/read_lines; grep+list; archive reversible + delete final; move) -- root=%s" % root)


if __name__ == "__main__":
    _selftest()
