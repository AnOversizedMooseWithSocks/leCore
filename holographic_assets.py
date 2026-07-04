"""holographic_assets.py -- keep track of EXTERNAL files (textures, models, ...) and repair their paths when they move.

THE PROBLEM (every 3-D pipeline has it)
A scene points at files on disk: project/textures/water/splashes/wave.png, project/models/boat.obj, and so on. Then
someone moves the project folder, or renames a parent directory, and every reference breaks at once. Fixing them one
by one is miserable. This module fixes them the way a person would reason about it:

  * RELOCATE FROM ONE. You re-point ONE broken asset to its new home. We compare its old and new paths, notice which
    trailing folders stayed the same (the preserved relative structure) and which parent changed (the move), and then
    re-point every OTHER broken asset that moved the same way. Move Documents/project -> Projects/project and re-point
    a single texture, and all the rest are found.
  * SEARCH BY STRUCTURE. If the clean prefix-swap doesn't land (things got reorganised a bit), we can SEARCH a folder
    for a file whose trailing folders/name match what we expect -- iterate down until the relative structure matches.
  * DETECT CHANGES. Each asset remembers a fingerprint (size + modified-time, and optionally a content hash), so we
    can tell when an external file has been edited on disk and needs re-importing.

DISTRIBUTED (the honestly-tricky part -- see the note on AssetLibrary.resolve): absolute paths are machine-specific, so
across machines we lean on TWO portable ideas -- a path RELATIVE to a project root, and a CONTENT HASH that identifies a
file no matter where it sits. Given a hash you can find the file by content under any search root (content-addressed),
which is the robust fallback when paths differ per machine.

Readable + stdlib only (os, hashlib, json, time). The path logic is POSIX-tested; Windows drive letters are handled
simply (a kept caveat, noted below).
"""
import os
import hashlib
import json
import time


# ---------------------------------------------------------------------------------------------------------
# Pure path helpers -- these do the relocation reasoning and are easy to test on their own (no disk needed).
# ---------------------------------------------------------------------------------------------------------
def _components(path):
    """Split a path into (is_absolute, [folder, folder, ..., name]). '/a/b/c.png' -> (True, ['a','b','c.png']).
    We normalise separators to '/' so the same logic works whether the path used '/' or '\\'."""
    norm = os.path.normpath(path).replace("\\", "/")
    isabs = norm.startswith("/") or (len(norm) > 1 and norm[1] == ":")   # POSIX root, or a Windows 'C:' drive
    comps = [c for c in norm.split("/") if c != ""]
    return isabs, comps


def _join(isabs, comps):
    """Rebuild a path from (is_absolute, components) -- the inverse of _components for our purposes."""
    body = "/".join(comps)
    return ("/" + body) if (isabs and not (comps and comps[0].endswith(":"))) else body


def common_suffix_len(a, b):
    """How many trailing components two component-lists share -- the length of the PRESERVED relative structure."""
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def relocation(old_path, new_path):
    """Work out the MOVE that turns old_path into new_path. Returns (old_prefix, new_prefix, preserved) where
    `preserved` is the shared trailing structure (the folders/name that stayed the same) and old_prefix/new_prefix are
    the parts before it that differ -- i.e. 'the folder moved from old_prefix to new_prefix'. Example:
        old = /Users/me/Documents/project/textures/water/wave.png
        new = /Users/me/Projects/project/textures/water/wave.png
      -> old_prefix=/Users/me/Documents, new_prefix=/Users/me/Projects,
         preserved=[project, textures, water, wave.png]."""
    ai, a = _components(old_path)
    bi, b = _components(new_path)
    k = common_suffix_len(a, b)
    preserved = a[len(a) - k:] if k else []
    old_prefix = _join(ai, a[:len(a) - k])
    new_prefix = _join(bi, b[:len(b) - k])
    return old_prefix, new_prefix, preserved


def under_prefix(path, prefix):
    """Is `path` inside `prefix`, compared FOLDER BY FOLDER (so /a/bc is NOT under /a/b)? Returns the components of
    `path` BELOW the prefix if so, else None."""
    _, p = _components(path)
    _, pre = _components(prefix)
    if len(pre) > len(p):
        return None
    if p[:len(pre)] != pre:
        return None
    return p[len(pre):]


def apply_relocation(path, old_prefix, new_prefix):
    """If `path` sits under old_prefix, return the equivalent path under new_prefix; else None. This is the one-line
    'the parent moved' rewrite applied to every other asset."""
    below = under_prefix(path, old_prefix)
    if below is None:
        return None
    nabs, npre = _components(new_prefix)
    return _join(nabs, npre + below)


def find_by_relative_tail(root, tail_components, min_match=1):
    """Search `root` (recursively) for files whose TRAILING path components match `tail_components`. Returns a list of
    (matched_path, how_many_trailing_components_matched), best matches first. This is the 'iterate down the folders
    until the relative structure lines up' search -- used when a clean prefix-swap doesn't find the file. `min_match`
    is the fewest trailing components that must line up (1 = match on the file name alone)."""
    tail = [c for c in tail_components if c]
    if not tail:
        return []
    hits = []
    want_name = tail[-1]
    for dirpath, _dirs, files in os.walk(root):
        if want_name not in files:
            continue                                          # cheap filter: the file name must be present here
        cand = os.path.join(dirpath, want_name)
        _, cparts = _components(cand)
        # how many trailing components line up between the candidate and what we expect?
        m = common_suffix_len(cparts, tail)
        if m >= min_match:
            hits.append((cand, m))
    hits.sort(key=lambda t: -t[1])                            # deepest (most-specific) structural match first
    return hits


# ---------------------------------------------------------------------------------------------------------
# Fingerprints -- how we know a file exists and whether it has changed.
# ---------------------------------------------------------------------------------------------------------
def file_hash(path, chunk=1 << 20):
    """The sha256 of a file's contents (streamed, so a big texture doesn't blow up memory). This is the portable,
    location-independent IDENTITY of a file -- the same bytes hash the same on any machine."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path, with_hash=False):
    """A small record of a file's state: whether it exists, its size and modified-time, and (optionally) its content
    hash. size+mtime is the CHEAP change check; the hash is the DEFINITIVE one (slower, opt-in for big files)."""
    if not os.path.exists(path):
        return {"exists": False, "size": None, "mtime": None, "sha256": None}
    st = os.stat(path)
    return {"exists": True, "size": st.st_size, "mtime": st.st_mtime,
            "sha256": (file_hash(path) if with_hash else None)}


# ---------------------------------------------------------------------------------------------------------
# The asset reference + the library that manages a whole scene's worth of them.
# ---------------------------------------------------------------------------------------------------------
class AssetRef:
    """One external file the scene depends on: its last-known `path`, an optional `role` label ('diffuse texture'),
    and the fingerprint we recorded the last time we looked. `id` is a stable handle for UIs/agents."""

    def __init__(self, path, role=None, fp=None, id=None):
        self.path = path
        self.role = role
        self.fp = fp or {}                                    # exists/size/mtime/sha256 as of last refresh
        self.id = id or hashlib.sha256(path.encode()).hexdigest()[:12]

    def exists(self):
        return os.path.exists(self.path)

    def refresh(self, with_hash=False):
        """Re-read the file's fingerprint from disk and store it (call after you've acknowledged a change). If this
        asset was already being tracked WITH a content hash, we keep hashing it -- so relinking a moved file doesn't
        silently drop the identity that distributed resolve relies on."""
        keep_hash = with_hash or bool(self.fp.get("sha256"))
        self.fp = fingerprint(self.path, with_hash=keep_hash)
        return self

    def status(self, with_hash=False):
        """'missing' if the file is gone, 'modified' if size/mtime (or hash, if we have one) differ from what we
        recorded, else 'ok'. This is how you tell an external file was edited on disk."""
        cur = fingerprint(self.path, with_hash=(with_hash and bool(self.fp.get("sha256"))))
        if not cur["exists"]:
            return "missing"
        if not self.fp:
            return "ok"                                       # nothing recorded to compare against yet
        if self.fp.get("sha256") and cur.get("sha256"):
            return "ok" if cur["sha256"] == self.fp["sha256"] else "modified"
        if cur["size"] != self.fp.get("size") or cur["mtime"] != self.fp.get("mtime"):
            return "modified"
        return "ok"

    def as_dict(self):
        return {"id": self.id, "path": self.path, "role": self.role, "fp": self.fp}


class AssetLibrary:
    """The scene's external files, with relocation + change detection. Add refs, ask which are missing(), fix them all
    by relinking ONE, or search a folder for the rest. Serialises to JSON as a portable manifest."""

    def __init__(self):
        self.assets = []

    # -- building ------------------------------------------------------------------------------------------
    def add(self, path, role=None, with_hash=False):
        """Register an external file. Records its fingerprint now (so later we can tell if it changed). Returns the ref."""
        ref = AssetRef(path, role=role, fp=fingerprint(path, with_hash=with_hash))
        self.assets.append(ref)
        return ref

    def get(self, ref_or_id_or_path):
        """Resolve a ref by object, id, or exact path."""
        if isinstance(ref_or_id_or_path, AssetRef):
            return ref_or_id_or_path
        for a in self.assets:
            if a.id == ref_or_id_or_path or a.path == ref_or_id_or_path:
                return a
        return None

    # -- inspection ----------------------------------------------------------------------------------------
    def missing(self):
        """The refs whose files are not where we last saw them."""
        return [a for a in self.assets if not a.exists()]

    def present(self):
        return [a for a in self.assets if a.exists()]

    def changed(self, with_hash=False):
        """The refs whose external file has been MODIFIED on disk since we recorded it (size/mtime, or hash if stored).
        These are the ones you'd want to re-import."""
        return [a for a in self.assets if a.status(with_hash=with_hash) == "modified"]

    def report(self, with_hash=False):
        """A UI/agent-friendly summary: counts + each asset's status."""
        rows = [{"id": a.id, "path": a.path, "role": a.role, "status": a.status(with_hash=with_hash)}
                for a in self.assets]
        counts = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {"counts": counts, "assets": rows}

    # -- the main event: relocation ------------------------------------------------------------------------
    def relink(self, ref_or_id_or_path, new_path, search=True, with_hash=False):
        """Re-point ONE asset to `new_path`, then automatically re-find every OTHER missing asset that moved the same
        way. Two passes: (1) PREFIX SWAP -- work out the parent that moved (old_prefix -> new_prefix) and rewrite the
        others under it; (2) if `search`, for anything still missing, SEARCH under the new parent for a file whose
        trailing folders/name match. Returns a report of what got relinked and what's still missing."""
        ref = self.get(ref_or_id_or_path)
        if ref is None:
            raise KeyError("no such asset: %r" % (ref_or_id_or_path,))
        if not os.path.exists(new_path):
            raise FileNotFoundError("the new location does not exist: %s" % new_path)

        old_prefix, new_prefix, preserved = relocation(ref.path, new_path)
        relinked = []

        # the asset we were handed
        ref.path = new_path
        ref.refresh(with_hash=with_hash)
        relinked.append({"id": ref.id, "to": new_path, "how": "provided"})

        # pass 1: prefix swap for every other missing asset that lived under the old parent
        for a in self.assets:
            if a is ref or a.exists():
                continue
            cand = apply_relocation(a.path, old_prefix, new_prefix)
            if cand and os.path.exists(cand):
                a.path = cand
                a.refresh(with_hash=with_hash)
                relinked.append({"id": a.id, "to": cand, "how": "prefix-swap"})

        # pass 2: structural search under the new parent for whatever is still missing
        if search and os.path.isdir(new_prefix):
            for a in self.assets:
                if a.exists():
                    continue
                _, tail = _components(a.path)
                hits = find_by_relative_tail(new_prefix, tail, min_match=1)
                if hits:
                    a.path = hits[0][0]
                    a.refresh(with_hash=with_hash)
                    relinked.append({"id": a.id, "to": hits[0][0], "how": "search(depth=%d)" % hits[0][1]})

        return {"relinked": relinked, "still_missing": [a.path for a in self.missing()],
                "moved_from": old_prefix, "moved_to": new_prefix}

    def search_under(self, root, with_hash=False):
        """Find missing assets by SEARCHING a folder (e.g. you just point at the relocated project folder). For each
        missing ref, look under `root` for a file whose trailing structure matches; if a content hash was recorded,
        prefer an exact content match (location-independent). Returns a relink report."""
        relinked = []
        for a in self.missing():
            chosen = None
            how = None
            # content-addressed first (portable across machines) if we know the hash
            want = a.fp.get("sha256")
            if want:
                found = find_by_hash(root, want)
                if found:
                    chosen, how = found, "content-hash"
            if chosen is None:                                # else fall back to structural match
                _, tail = _components(a.path)
                hits = find_by_relative_tail(root, tail, min_match=1)
                if hits:
                    chosen, how = hits[0][0], "search(depth=%d)" % hits[0][1]
            if chosen:
                a.path = chosen
                a.refresh(with_hash=with_hash)
                relinked.append({"id": a.id, "to": chosen, "how": how})
        return {"relinked": relinked, "still_missing": [a.path for a in self.missing()]}

    # -- distributed / portability -------------------------------------------------------------------------
    def resolve(self, ref_or_id_or_path, roots=()):
        """Return a USABLE local path for an asset, trying in order: the stored path (if it exists); a match by
        CONTENT HASH under any of `roots` (location-independent -- the robust distributed fallback); a structural
        match under `roots`. Returns the path or None. This is the seam for a distributed setup: give each machine its
        own `roots`, reference assets by hash, and resolution finds them wherever they landed."""
        ref = self.get(ref_or_id_or_path)
        if ref is None:
            return None
        if ref.exists():
            return ref.path
        want = ref.fp.get("sha256")
        for root in roots:
            if want:
                f = find_by_hash(root, want)
                if f:
                    return f
            _, tail = _components(ref.path)
            hits = find_by_relative_tail(root, tail, min_match=1)
            if hits:
                return hits[0][0]
        return None

    def add_hashes(self):
        """Compute + store content hashes for every present asset (so distributed resolve/relink can use them). Slower;
        do it once when you're ready to make the manifest portable."""
        for a in self.present():
            a.refresh(with_hash=True)
        return self

    # -- persistence (the manifest) ------------------------------------------------------------------------
    def to_dict(self):
        return {"version": 1, "assets": [a.as_dict() for a in self.assets]}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d):
        lib = cls()
        for a in d.get("assets", []):
            lib.assets.append(AssetRef(a["path"], role=a.get("role"), fp=a.get("fp"), id=a.get("id")))
        return lib

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        return "AssetLibrary(%d assets, %d missing)" % (len(self.assets), len(self.missing()))


def find_by_hash(root, target_hash):
    """Find a file under `root` whose contents hash to `target_hash` -- content-addressed lookup, so it works even if
    the path is completely different. Returns the first match's path, or None. (Hashes every file, so scope `root`.)"""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            try:
                if file_hash(p) == target_hash:
                    return p
            except OSError:
                continue
    return None


def _selftest():
    import tempfile
    import shutil

    root = tempfile.mkdtemp(prefix="lecore_assets_")
    try:
        # build an OLD project tree with a few assets in a folder hierarchy
        old = os.path.join(root, "Documents", "project")
        for rel in ("textures/water/splashes/wave.png", "textures/stone/wall.png", "models/boat.obj"):
            p = os.path.join(old, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(("data:" + rel).encode())

        lib = AssetLibrary()
        for rel in ("textures/water/splashes/wave.png", "textures/stone/wall.png", "models/boat.obj"):
            lib.add(os.path.join(old, *rel.split("/")), role=rel)
        assert len(lib.missing()) == 0

        # MOVE the whole project (Documents/project -> Projects/project) without changing anything inside
        new = os.path.join(root, "Projects", "project")
        os.makedirs(os.path.dirname(new), exist_ok=True)
        shutil.move(old, new)
        assert len(lib.missing()) == 3, "all three should be missing after the move"

        # RELOCATE FROM ONE: re-point a single texture; the other two should be found automatically
        one_new = os.path.join(new, "textures", "water", "splashes", "wave.png")
        rep = lib.relink(lib.assets[0].path, one_new)
        assert len(lib.missing()) == 0, ("still missing: %s" % rep["still_missing"])
        hows = sorted(r["how"].split("(")[0] for r in rep["relinked"])
        assert "provided" in hows and "prefix-swap" in hows, hows

        # SEARCH BY STRUCTURE: a file that got reorganised (moved up a level) is still found by its trailing name.
        # Use an ISOLATED file here so we don't disturb `lib`'s tracked assets.
        struct_old = os.path.join(new, "extra", "props", "barrel.obj")
        os.makedirs(os.path.dirname(struct_old), exist_ok=True)
        with open(struct_old, "wb") as f:
            f.write(b"barrel")
        lib2 = AssetLibrary()
        lib2.add(struct_old, role="model")
        shutil.move(struct_old, os.path.join(root, "Projects", "barrel.obj"))   # reorganise: clean prefix-swap won't hit
        found = lib2.search_under(os.path.join(root, "Projects"))
        assert len(lib2.missing()) == 0, found                # found by trailing-name search even though it moved

        # DETECT CHANGES: edit a file on disk -> status flips to 'modified'
        wall = lib.get(lib.assets[1].id)
        assert wall.status() == "ok"
        time.sleep(0.01)
        with open(wall.path, "ab") as f:
            f.write(b" edited")
        os.utime(wall.path, None)
        assert wall.status() == "modified", "an edited file should read as modified"
        wall.refresh()
        assert wall.status() == "ok", "after refresh it's acknowledged"

        # DISTRIBUTED: content-hash find works even when the path is totally different
        lib.add_hashes()
        target = lib.get(lib.assets[2].id)                    # boat.obj
        want = target.fp["sha256"]
        elsewhere = os.path.join(root, "SomewhereElse", "renamed.obj")
        os.makedirs(os.path.dirname(elsewhere), exist_ok=True)
        shutil.copy(target.path, elsewhere)
        assert find_by_hash(os.path.join(root, "SomewhereElse"), want) == elsewhere

        # the manifest round-trips
        d = lib.to_dict()
        lib3 = AssetLibrary.from_dict(d)
        assert len(lib3.assets) == len(lib.assets)

        print("OK: holographic_assets self-test passed (moved a project + relinked ALL 3 from ONE; structural search "
              "found a reorganised file by name; edited file detected as 'modified' then acknowledged; content-hash "
              "found a renamed copy; manifest round-trips)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
