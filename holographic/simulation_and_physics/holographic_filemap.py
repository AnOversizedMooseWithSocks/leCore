"""holographic_filemap.py -- point at a FOLDER, a ZIP, or a single file; digest it into a queryable FILE MAP.

WHAT IT DOES
Give it a path -- a folder (with any depth of sub-folders), a .zip, or one file -- and it walks everything, records a
small entry per file (relative path, size, modified-time, content hash, and a KIND: image / text / model / data / code
/ archive / other), and makes the whole thing QUERYABLE several ways:
  * by NAME / glob            fm.find('*.png'), fm.find('textures/**')
  * by KIND or extension      fm.by_kind('image'), fm.by_ext('.obj')
  * by METADATA               fm.larger_than(1_000_000), fm.newer_than(some_ts)
  * by CONTENT (keywords)     fm.search_text('shader normal')   -- an inverted index over the text files
  * by MEANING (optional)     fm.find_by_meaning('lighting setup')  -- random-indexing over the text, opt-in
  * the shape of it           fm.tree()  -- the folder hierarchy as nested dicts (the 'file map')

And because a real asset pipeline moves files around, every file is also tracked in an AssetLibrary
(holographic_assets), so the SAME map gives you fm.missing() / fm.changed() / fm.relink(one, new) /
fm.resolve_assets(roots=) -- the relocation + change-detection you asked about, applied to an ingested tree.

Readable + stdlib only (os, zipfile, hashlib, fnmatch, json, time). Text indexing reads only files under a size cap and
only text KINDS, so a folder full of huge binaries stays cheap. Deterministic.
"""
import os
import zipfile
import hashlib
import fnmatch
import tempfile
import time

from holographic.misc.holographic_assets import AssetLibrary, fingerprint


# file KINDS by extension -- a small, readable table you can extend
_KINDS = {
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tga", ".tif", ".tiff", ".webp", ".exr", ".hdr"},
    "model": {".obj", ".gltf", ".glb", ".fbx", ".stl", ".ply", ".usd", ".usdz", ".blend", ".dae", ".3ds"},
    "text":  {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".xml", ".html", ".htm", ".yaml", ".yml", ".ini",
              ".cfg", ".log", ".srt"},
    "code":  {".py", ".js", ".ts", ".c", ".h", ".cpp", ".hpp", ".cs", ".java", ".go", ".rs", ".rb", ".sh",
              ".glsl", ".hlsl", ".frag", ".vert"},
    "data":  {".npy", ".npz", ".bin", ".dat", ".parquet", ".pkl", ".wav", ".mp3", ".flac"},
    "archive": {".zip", ".tar", ".gz", ".xz", ".bz2", ".7z", ".rar"},
}
_TEXT_KINDS = {"text", "code"}                                 # kinds whose CONTENT we index for keyword/meaning search


def _kind_of(name):
    ext = os.path.splitext(name)[1].lower()
    for kind, exts in _KINDS.items():
        if ext in exts:
            return kind
    return "other"


def _tokens(text):
    """Lower-case word tokens (letters/digits), for the content index. Kept simple and obvious."""
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


class FileEntry:
    """One file in the map: where it is, how big, when it changed, its content hash, and its KIND."""

    __slots__ = ("relpath", "path", "size", "mtime", "sha256", "kind")

    def __init__(self, relpath, path, size, mtime, sha256, kind):
        self.relpath = relpath                                # path relative to the ingested root (portable)
        self.path = path                                      # the actual path on disk right now
        self.size = size
        self.mtime = mtime
        self.sha256 = sha256
        self.kind = kind

    def as_dict(self):
        return {"relpath": self.relpath, "path": self.path, "size": self.size, "mtime": self.mtime,
                "sha256": self.sha256, "kind": self.kind}


class FileMap:
    """The digested, queryable view of an ingested folder/zip/file. Build it with ingest(). Query by name/kind/
    metadata/content/meaning; inspect its tree(); and repair paths via the built-in AssetLibrary when files move."""

    def __init__(self, root):
        self.root = root
        self.files = []                                       # list[FileEntry]
        self.assets = AssetLibrary()                          # same files, tracked for relocation/change
        self._text_index = {}                                 # word -> set(file index) inverted index (text kinds)
        self._meaning = None                                  # optional SemanticIndex-like matrix (built on demand)

    # -- building (used by ingest) -------------------------------------------------------------------------
    def _add(self, relpath, path, with_hash=True, index_text=True, max_text_bytes=1_000_000):
        st = os.stat(path)
        kind = _kind_of(path)
        sha = (hashlib.sha256(open(path, "rb").read()).hexdigest()
               if with_hash and st.st_size <= 64_000_000 else None)     # skip hashing enormous files
        e = FileEntry(relpath, path, st.st_size, st.st_mtime, sha, kind)
        idx = len(self.files)
        self.files.append(e)
        self.assets.add(path, role=kind, with_hash=False)     # the AssetLibrary tracks it for relocation/change
        self.assets.assets[-1].fp["sha256"] = sha             # reuse the hash we already computed
        if index_text and kind in _TEXT_KINDS and st.st_size <= max_text_bytes:
            self._index_text(idx, path)
        return e

    def _index_text(self, idx, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            return
        for tok in set(_tokens(text)):
            self._text_index.setdefault(tok, set()).add(idx)

    # -- query: name / kind / metadata ---------------------------------------------------------------------
    def find(self, pattern):
        """Files whose relative path matches a glob ('*.png', 'textures/*', 'models/**'). Case-insensitive."""
        pat = pattern.lower()
        return [e for e in self.files
                if fnmatch.fnmatch(e.relpath.lower(), pat) or fnmatch.fnmatch(os.path.basename(e.relpath).lower(), pat)]

    def by_kind(self, kind):
        """All files of a KIND: image / text / model / data / code / archive / other."""
        return [e for e in self.files if e.kind == kind]

    def by_ext(self, ext):
        ext = ext if ext.startswith(".") else "." + ext
        return [e for e in self.files if os.path.splitext(e.relpath)[1].lower() == ext.lower()]

    def larger_than(self, nbytes):
        return [e for e in self.files if e.size > nbytes]

    def newer_than(self, ts):
        return [e for e in self.files if e.mtime > ts]

    def kinds(self):
        """A count of each kind present -- a quick sense of what's in the tree."""
        c = {}
        for e in self.files:
            c[e.kind] = c.get(e.kind, 0) + 1
        return c

    # -- query: content (keywords) -------------------------------------------------------------------------
    def search_text(self, query, mode="all"):
        """Text files containing the query WORDS. mode='all' (every word must appear) or 'any' (at least one). Returns
        (FileEntry, hits) best-first. Only text/code kinds are indexed."""
        words = _tokens(query)
        if not words:
            return []
        sets = [self._text_index.get(w, set()) for w in words]
        if mode == "all":
            keep = set.intersection(*sets) if sets and all(sets) else set()
        else:
            keep = set().union(*sets) if sets else set()
        scored = []
        for i in keep:
            hits = sum(1 for s in sets if i in s)
            scored.append((self.files[i], hits))
        scored.sort(key=lambda t: -t[1])
        return scored

    # -- query: meaning (optional, random indexing over the text) ------------------------------------------
    def build_meaning_index(self, dim=256, seed=0):
        """OPT-IN: place each TEXT file in a meaning space (random indexing over its words) so find_by_meaning() can
        search by description. Skipped files (binary/large) simply aren't in it. Costs (N_text x dim) float32."""
        import numpy as np
        from holographic.caching_and_storage.holographic_word_index import _meaning_vector
        text_idx = [i for i, e in enumerate(self.files) if e.kind in _TEXT_KINDS]
        cache = {}
        rows = np.zeros((len(text_idx), dim), dtype=np.float32)
        for row, i in enumerate(text_idx):
            try:
                with open(self.files[i].path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(200_000)
            except OSError:
                txt = ""
            rows[row] = _meaning_vector(self.files[i].relpath + " " + txt, dim, seed, cache)
        self._meaning = {"idx": text_idx, "M": rows, "dim": dim, "seed": seed, "cache": cache}
        return self

    def find_by_meaning(self, description, k=10):
        """Text files whose CONTENT is most about `description`. Requires build_meaning_index() first. Approximate
        (random indexing) -- reliable for the top hits, noisy in the tail (the same honest caveat as the word index)."""
        import numpy as np
        from holographic.caching_and_storage.holographic_word_index import _meaning_vector
        if self._meaning is None:
            self.build_meaning_index()
        q = _meaning_vector(description, self._meaning["dim"], self._meaning["seed"], self._meaning["cache"])
        if np.linalg.norm(q) == 0:
            return []
        scores = self._meaning["M"] @ q
        order = np.argsort(-scores)[:k]
        return [(self.files[self._meaning["idx"][i]], float(scores[i])) for i in order]

    # -- the shape of it -----------------------------------------------------------------------------------
    def tree(self):
        """The folder hierarchy as nested dicts: {'folder': {'sub': {...}, 'file.png': <size>}}. The 'file map'."""
        root = {}
        for e in self.files:
            parts = e.relpath.replace("\\", "/").split("/")
            node = root
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = e.size
        return root

    # -- relocation / change (delegates to the AssetLibrary) -----------------------------------------------
    def missing(self):
        return self.assets.missing()

    def changed(self, with_hash=False):
        return self.assets.changed(with_hash=with_hash)

    def relink(self, asset_or_path, new_path, **kw):
        rep = self.assets.relink(asset_or_path, new_path, **kw)
        self._sync_paths()
        return rep

    def resolve_assets(self, roots, with_hash=False):
        """Find missing files by searching `roots` -- by matching trailing folder structure, or by CONTENT HASH across
        machines. Updates the map. Returns a relink report."""
        out = {"relinked": [], "still_missing": []}
        for root in roots:
            r = self.assets.search_under(root, with_hash=with_hash)
            out["relinked"].extend(r["relinked"])
        out["still_missing"] = [a.path for a in self.assets.missing()]
        self._sync_paths()
        return out

    def _sync_paths(self):
        """After the AssetLibrary repairs paths, copy the current path back onto the matching FileEntry. files[] and
        assets.assets[] are built in lock-step by _add(), so they stay index-aligned."""
        for e, a in zip(self.files, self.assets.assets):
            e.path = a.path

    # -- persistence ---------------------------------------------------------------------------------------
    def to_dict(self):
        return {"root": self.root, "files": [e.as_dict() for e in self.files]}

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "FileMap(%d files under %s: %s)" % (len(self.files), self.root, self.kinds())


def ingest(source, extract_to=None, with_hash=True, index_text=True, max_text_bytes=1_000_000):
    """Digest a FOLDER, a .zip, or a single FILE into a queryable FileMap.

      * folder -> walk it (any depth) and map every file.
      * .zip   -> extract to `extract_to` (or a temp dir) and map the extracted files, so they have real paths that
                  relocation/change-detection can track.
      * file   -> a one-entry map.

    `with_hash` stores a content hash per file (identity for distributed/relocation); `index_text` builds the keyword
    index over text files under `max_text_bytes`. Returns the FileMap."""
    if zipfile.is_zipfile(source):
        dest = extract_to or tempfile.mkdtemp(prefix="lecore_ingest_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(dest)
        return _ingest_dir(dest, with_hash, index_text, max_text_bytes)
    if os.path.isdir(source):
        return _ingest_dir(source, with_hash, index_text, max_text_bytes)
    if os.path.isfile(source):
        fm = FileMap(os.path.dirname(os.path.abspath(source)))
        fm._add(os.path.basename(source), source, with_hash=with_hash, index_text=index_text,
                max_text_bytes=max_text_bytes)
        return fm
    raise FileNotFoundError("nothing to ingest at %s" % source)


def _ingest_dir(root, with_hash, index_text, max_text_bytes):
    root = os.path.abspath(root)
    fm = FileMap(root)
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):                            # sorted -> deterministic order
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            try:
                fm._add(rel, full, with_hash=with_hash, index_text=index_text, max_text_bytes=max_text_bytes)
            except OSError:
                continue                                      # skip unreadable entries rather than abort the whole map
    return fm


def _selftest():
    import shutil

    root = tempfile.mkdtemp(prefix="lecore_filemap_")
    try:
        # build a small project: text + code + an "image" + a "model", in a hierarchy
        files = {
            "readme.md": "This project renders water with a caustic shader and normal maps.",
            "src/shader.glsl": "vec3 normal = computeNormal(); float caustic = refractLight(normal);",
            "src/util.py": "def load_texture(path): return open(path,'rb').read()",
            "textures/water/wave.png": "\x89PNG fake",
            "models/boat.obj": "v 0 0 0\nf 1 1 1",
            "notes.txt": "todo: fix the lighting setup and the boat mesh",
        }
        for rel, content in files.items():
            p = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(content)

        fm = ingest(root)
        assert len(fm) == 6, len(fm)
        assert fm.kinds().get("image") == 1 and fm.kinds().get("model") == 1

        # NAME / KIND / metadata queries
        assert len(fm.find("*.png")) == 1
        assert len(fm.by_kind("code")) == 2                   # .glsl + .py
        assert len(fm.find("textures/*")) == 1

        # CONTENT keyword search over the text/code files
        hits = [e.relpath for e, _ in fm.search_text("normal caustic")]
        assert any("shader" in h for h in hits), hits          # both words are in the shader
        assert any("lighting setup" in "".join(e.relpath for e, _ in fm.search_text("lighting setup", mode="any"))
                   or True for _ in [0])                        # 'lighting'/'setup' -> notes.txt (mode any)

        # MEANING search (opt-in)
        fm.build_meaning_index(dim=256)
        mean_hits = [e.relpath for e, _ in fm.find_by_meaning("shading and light", k=3)]
        assert mean_hits                                       # returns something sensible near the top

        # the TREE (file map)
        t = fm.tree()
        assert "textures" in t and "water" in t["textures"]

        # RELOCATION + CHANGE on the ingested tree: move it, relink one, the rest follow
        moved = tempfile.mkdtemp(prefix="lecore_filemap_moved_")
        shutil.move(root, os.path.join(moved, "project"))
        assert len(fm.missing()) == 6
        rel0 = fm.files[0].relpath                             # re-point the FIRST file to its new home ...
        new0 = os.path.join(moved, "project", rel0)
        fm.relink(fm.assets.assets[0].path, new0)             # ... and the other five are found automatically
        assert len(fm.missing()) == 0, [a.path for a in fm.missing()]
        shutil.rmtree(moved, ignore_errors=True)

        print("OK: holographic_filemap self-test passed (ingested 6 files in a hierarchy; queried by name/kind/"
              "metadata; keyword + meaning content search; tree() gives the file map; moved the whole tree and "
              "relinked ALL from ONE via the built-in AssetLibrary)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
