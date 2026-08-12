"""VAULT -- a local, linked, markdown knowledge base the MODEL can use.

Obsidian's actual core is small and worth copying exactly: plain markdown files
on disk, `[[wikilinks]]` between them, backlinks derived automatically, tags,
aliases, and a graph you can inspect for clusters and orphans. Everything else
is UI. The files are the product; if this engine disappears the notes are still
readable in any editor -- and an existing Obsidian vault can be opened here
directly, because the format is not ours.

WHAT MAKES THIS DIFFERENT FROM A NOTE APP: the model is a first-class user of
it. The corpus resident grounds answers in vault notes (retrieval into the
residual stream, no context window spent), the fact checker builds evidence from
the same notes, and residents WRITE notes of their own with provenance -- so an
inner conclusion becomes a linked note that later retrieval can find. A human
and a swarm keep the same notebook.

BACKLINKS ARE DERIVED, NEVER STORED. A stored backlink is a second copy of a
fact that can disagree with the first; the links live in the text, and the
reverse index is computed. Rename a note and the graph is recomputed rather than
migrated.
"""

import json
import os
import re
import time


WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
TAG = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def slug(title):
    """A filename that survives every filesystem, without losing the title."""
    keep = "".join(c if (c.isalnum() or c in " -_") else " " for c in str(title))
    return "-".join(keep.split()).strip("-").lower() or "untitled"


class Note:
    """One markdown file: frontmatter, body, and what it points at."""

    def __init__(self, path, title, body, meta=None):
        self.path = path
        self.title = title
        self.body = body
        self.meta = meta or {}

    @property
    def links(self):
        """Outgoing [[wikilinks]], by target title (aliases resolved by Vault)."""
        return [m.group(1).strip() for m in WIKILINK.finditer(self.body)]

    @property
    def tags(self):
        inline = {m.group(1) for m in TAG.finditer(self.body)}
        front = self.meta.get("tags") or []
        if isinstance(front, str):
            front = [t.strip() for t in front.split(",") if t.strip()]
        return sorted(inline | set(front))

    @property
    def aliases(self):
        al = self.meta.get("aliases") or []
        if isinstance(al, str):
            al = [a.strip() for a in al.split(",") if a.strip()]
        return list(al)

    def text(self):
        """Title plus body -- what retrieval and evidence actually see."""
        return "%s\n\n%s" % (self.title, self.body)


class Vault:
    """A folder of markdown notes with links, backlinks, tags and a graph."""

    def __init__(self, root):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)

    # ---- reading ----

    def paths(self):
        out = []
        for base, _dirs, files in os.walk(self.root):
            if os.path.basename(base).startswith("."):
                continue
            for f in sorted(files):
                if f.endswith(".md"):
                    out.append(os.path.join(base, f))
        return out

    def _parse(self, path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        meta = {}
        m = FRONTMATTER.match(raw)
        body = raw
        if m:
            body = raw[m.end():]
            for line in m.group(1).split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        v = [x.strip().strip("'\"") for x in v[1:-1].split(",")
                             if x.strip()]
                    meta[k.strip()] = v
        title = meta.get("title") or os.path.basename(path)[:-3]
        return Note(path, title, body, meta)

    def notes(self):
        return [self._parse(p) for p in self.paths()]

    def get(self, name):
        """Find by title, alias, or slug -- the three ways a link can spell it."""
        want = str(name).strip().lower()
        for n in self.notes():
            if n.title.lower() == want or slug(n.title) == slug(want):
                return n
            if any(a.lower() == want for a in n.aliases):
                return n
        return None

    # ---- writing ----

    def write(self, title, body, tags=(), aliases=(), author=None, kind=None,
              append=False):
        """Create or update a note. Frontmatter records provenance, so a note
        written by the swarm is never mistaken for one a person wrote."""
        path = os.path.join(self.root, slug(title) + ".md")
        meta = {"title": str(title), "updated": time.strftime("%Y-%m-%d %H:%M")}
        if tags:
            meta["tags"] = list(tags)
        if aliases:
            meta["aliases"] = list(aliases)
        if author:
            meta["author"] = str(author)
        if kind:
            meta["kind"] = str(kind)
        old = ""
        if append and os.path.exists(path):
            prev = self._parse(path)
            old = prev.body.rstrip() + "\n\n"
            for k, v in prev.meta.items():
                meta.setdefault(k, v)
        lines = ["---"]
        for k, v in meta.items():
            lines.append("%s: %s" % (k, json.dumps(v) if isinstance(v, list) else v))
        lines.append("---")
        text = "\n".join(lines) + "\n" + old + str(body).strip() + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def daily(self, body, tags=()):
        """Today's note, appended to -- the habit that makes a vault accumulate."""
        return self.write(time.strftime("%Y-%m-%d"), body,
                          tags=tuple(tags) + ("daily",), append=True)

    # ---- structure ----

    def graph(self):
        """Nodes, edges, backlinks, orphans and unresolved links.

        Backlinks are DERIVED here rather than stored: a stored reverse index is
        a second copy that can disagree with the text, and the text is the
        product."""
        notes = self.notes()
        by_key = {}
        for n in notes:
            by_key[n.title.lower()] = n.title
            by_key[slug(n.title)] = n.title
            for a in n.aliases:
                by_key[a.lower()] = n.title
        edges, unresolved = [], []
        backlinks = {n.title: [] for n in notes}
        for n in notes:
            for target in n.links:
                key = target.strip().lower()
                real = by_key.get(key) or by_key.get(slug(target))
                if real is None:
                    unresolved.append((n.title, target))
                    continue
                edges.append((n.title, real))
                backlinks.setdefault(real, []).append(n.title)
        linked = {a for a, _b in edges} | {b for _a, b in edges}
        orphans = sorted(n.title for n in notes if n.title not in linked)
        return {"nodes": sorted(n.title for n in notes), "edges": edges,
                "backlinks": {k: sorted(set(v)) for k, v in backlinks.items()},
                "orphans": orphans, "unresolved": unresolved,
                "tags": sorted({t for n in notes for t in n.tags})}

    def backlinks(self, title):
        return self.graph()["backlinks"].get(title, [])

    def clusters(self):
        """Connected components -- the honest version of a graph view for a
        terminal: which groups of notes actually hang together."""
        g = self.graph()
        adj = {n: set() for n in g["nodes"]}
        for a, b in g["edges"]:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        seen, out = set(), []
        for n in g["nodes"]:
            if n in seen:
                continue
            stack, comp = [n], []
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                stack.extend(adj.get(cur, ()))
            out.append(sorted(comp))
        return sorted(out, key=len, reverse=True)

    # ---- use ----

    def search(self, mind, query, top=3, tags=None):
        """Rank notes for a query, delegating to leCore's own BM25."""
        notes = [n for n in self.notes()
                 if tags is None or (set(tags) & set(n.tags))]
        if not notes:
            return []
        ranked = mind.bm25_rank(query, [n.text() for n in notes], top=int(top)) or []
        out = []
        for item in ranked:
            idx = item[0] if isinstance(item, (tuple, list)) else item
            n = notes[int(idx)] if isinstance(idx, int) else None
            if n is None:
                n = next((x for x in notes if x.text() == idx), None)
            if n is not None:
                out.append(n)
        return out

    def passages(self, max_chars=600):
        """The vault as a grounding corpus: one passage per paragraph, each
        carrying its note title so a retrieved fact can be traced home."""
        out = []
        for n in self.notes():
            for para in n.body.split("\n\n"):
                para = para.strip()
                if len(para) < 40:
                    continue
                out.append("%s: %s" % (n.title, para[:max_chars]))
        return out


def _selftest():
    import tempfile

    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    v = Vault(tempfile.mkdtemp())

    v.write("Delta Rule", "The delta rule updates a memory matrix toward a "
            "target, and is the core of [[Gated DeltaNet]]. #vsa",
            tags=("method",), aliases=("delta update",))
    v.write("Gated DeltaNet", "Gated DeltaNet decouples erase from write. "
            "It builds on the [[Delta Rule]] and appears in [[Qwen3.5]]. #arch")
    v.write("Qwen3.5", "A hybrid model with linear attention layers. #arch")
    v.write("Sourdough", "Bread from flour, water, salt and a wild starter.")

    # ---- links resolve, and BACKLINKS ARE DERIVED both ways ----
    g = v.graph()
    assert ("Delta Rule", "Gated DeltaNet") in g["edges"], g["edges"]
    assert "Delta Rule" in g["backlinks"]["Gated DeltaNet"]
    assert "Gated DeltaNet" in g["backlinks"]["Delta Rule"], g["backlinks"]
    assert g["orphans"] == ["Sourdough"], g["orphans"]     # linked to nothing
    assert set(g["tags"]) >= {"arch", "method", "vsa"}, g["tags"]

    # ---- an ALIAS is a real way to reach a note ----
    assert v.get("delta update").title == "Delta Rule"
    assert v.get("gated-deltanet").title == "Gated DeltaNet"

    # ---- a link to a note that does not exist is REPORTED, not swallowed ----
    v.write("Loose End", "This points at [[Nothing At All]].")
    assert ("Loose End", "Nothing At All") in v.graph()["unresolved"]

    # ---- clusters: the connected story separates from the unrelated note ----
    cl = v.clusters()
    assert set(cl[0]) == {"Delta Rule", "Gated DeltaNet", "Qwen3.5"}, cl
    assert ["Sourdough"] in cl

    # ---- the MODEL's side: retrieval finds the right note, and the vault is a
    #      grounding corpus whose passages carry their source title
    hit = v.search(mind, "erase write decouple")[0]
    assert hit.title == "Gated DeltaNet", hit.title
    assert v.search(mind, "flour water salt starter")[0].title == "Sourdough"
    ps = v.passages()
    assert any(p.startswith("Sourdough:") for p in ps), ps[:2]

    # ---- PROVENANCE: a note written by a resident says so, in the file ----
    v.write("Spectral Finding", "Only the decay gates showed a clear gap.",
            author="swarm", kind="note", tags=("spectra",))
    n = v.get("Spectral Finding")
    assert n.meta.get("author") == "swarm" and "spectra" in n.tags
    with open(n.path, encoding="utf-8") as f:
        raw = f.read()
    assert raw.startswith("---") and "author: swarm" in raw   # plain, portable

    # ---- append keeps the earlier text (a daily note must accumulate) ----
    v.daily("first entry")
    v.daily("second entry")
    today = v.get(time.strftime("%Y-%m-%d"))
    assert "first entry" in today.body and "second entry" in today.body

    print("vault selftest OK -- %d notes; wikilinks resolve, backlinks derived "
          "both ways, aliases and slugs both reach a note, unresolved links are "
          "reported not swallowed, clusters separate (%d), orphans found (%s), "
          "retrieval picks the right note, passages carry their source, "
          "provenance is in the file, and daily notes accumulate"
          % (len(v.notes()), len(v.clusters()), g["orphans"][0]))


if __name__ == "__main__":
    _selftest()
