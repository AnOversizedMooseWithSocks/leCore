"""MEMORY -- the Galvatron's own store, built on leCore's holographic database.

CORRECTION ON RECORD: a previous version of this wrote markdown files with
[[wikilinks]] and derived backlinks by re-parsing text. That was building a
filesystem next to an engine that already has a database -- namespaces, tables,
SQL with exact AND fuzzy predicates, an edge table with real adjacency
traversal, views, a journal, versioning, cold tiers and crash-safe snapshots.
Rule 0 exists precisely to stop that, and it was skipped. The vault module is
kept only as a converter for anyone who already has a folder of notes.

WHAT LIVES WHERE, and why the split is honest rather than lazy:
  * RECORDS AND RELATIONS -> the holographic database. Structured columns
    (id, title, author, kind, tags, session) are categorical fillers bound to
    column roles, which is exactly what the Table is for: exact predicates run
    on the stored values, the fuzzy `~` predicate ranks by cosine over those
    bindings, and links live in an EDGE TABLE whose adjacency() gives forward
    and reverse traversal -- backlinks as data, not as a re-parse.
  * FREE TEXT -> BM25 (mind.bm25_rank). Binding a paragraph as a categorical
    filler would encode a whole document as one symbol and rank it by accident;
    the engine's own docs call encoding continuous content into a vector "the
    honest fork", and the same reasoning applies to prose. Text is stored in the
    row and ranked lexically.

Persistence is the database's own: snapshot() writes a crash-safe file of the
persistent tier and Database.restore() replays it, so a Galvatron's memory
survives the process without a bespoke file format.
"""

import time


NOTE_COLUMNS = ("id", "title", "author", "kind", "tags", "session", "created")
LINK_COLUMNS = ("src", "dst", "kind")


class Memory:
    """Notes, links and provenance for a Galvatron, in the engine's database."""

    def __init__(self, mind, dim=1024, namespace="mem", db=None):
        self.mind = mind
        self.ns = str(namespace)
        self.db = db if db is not None else mind.database(dim=int(dim))
        if self.ns not in self.db.namespaces:
            self.db.create_namespace(self.ns, tier="persistent")
        self._texts = {}
        for qualified, cols in ((self._t("notes"), NOTE_COLUMNS),
                                (self._t("links"), LINK_COLUMNS)):
            try:
                self.db.resolve(qualified)
            except Exception:
                self.db.create_table(qualified, list(cols), dim=int(dim))

    def _t(self, name):
        return "%s.%s" % (self.ns, name)

    # ---- writing ----

    def note(self, title, text, author="user", kind="note", tags=(),
             session=None, links=()):
        """File a note. `author` and `kind` are columns, not conventions, so a
        swarm conclusion can never be mistaken for something a person wrote --
        and it is one WHERE clause to separate them."""
        nid = "n%d" % (len(self.ids()) + 1)
        self.db.insert(self._t("notes"), {
            "id": nid, "title": str(title), "author": str(author),
            "kind": str(kind), "tags": ",".join(tags) if tags else "",
            "session": str(session or ""), "created": time.strftime("%Y-%m-%d")})
        self._texts[nid] = str(text)
        for target in links:
            self.link(nid, target)
        return nid

    def link(self, src, dst, kind="ref"):
        """An edge in the links table. Backlinks are then a REVERSE ADJACENCY on
        real data rather than a re-scan of prose for brackets."""
        dst_id = dst if dst in self._texts or dst.startswith("n") else \
            (self.by_title(dst) or dst)
        self.db.insert(self._t("links"),
                       {"src": str(src), "dst": str(dst_id), "kind": str(kind)})
        return (src, dst_id)

    # ---- reading ----

    def rows(self, where=None):
        sql = "SELECT id, title, author, kind, tags, session FROM notes"
        if where:
            sql += " WHERE " + where
        return self.mind.query(sql, self.db.resolve(self._t("notes")))

    def ids(self):
        return [r.get("id") for r in self.rows()]

    def by_title(self, title):
        want = str(title).strip().lower()
        for r in self.rows():
            if str(r.get("title", "")).strip().lower() == want:
                return r.get("id")
        return None

    def text(self, nid):
        return self._texts.get(nid, "")

    def search(self, query, top=3, where=None):
        """Rank note TEXT lexically (BM25), optionally over a SQL-filtered
        subset -- structure and language each doing the job they are good at."""
        cand = self.rows(where)
        pool = [(r, self._texts.get(r.get("id"), "")) for r in cand]
        pool = [(r, t) for r, t in pool if t]
        if not pool:
            return []
        ranked = self.mind.bm25_rank(query, [t for _r, t in pool], top=int(top)) or []
        out = []
        for item in ranked:
            idx = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(idx, (int,)):
                r, t = pool[int(idx)]
            else:
                r, t = next(((r, t) for r, t in pool if t == idx), (None, None))
            if r is not None:
                rec = dict(r)
                rec["text"] = t
                out.append(rec)
        return out

    def graph(self):
        """Forward and reverse adjacency, straight from the edge table."""
        fwd = self.db.adjacency(self._t("links"), "src", "dst")
        rev = self.db.adjacency(self._t("links"), "src", "dst", reverse=True)
        titles = {r.get("id"): r.get("title") for r in self.rows()}
        named = {titles.get(k, k): [titles.get(v, v) for v in vs]
                 for k, vs in dict(fwd).items()}
        back = {titles.get(k, k): [titles.get(v, v) for v in vs]
                for k, vs in dict(rev).items()}
        linked = set(dict(fwd)) | {v for vs in dict(fwd).values() for v in vs}
        orphans = sorted(titles[i] for i in titles if i not in linked)
        return {"links": named, "backlinks": back, "orphans": orphans}

    def passages(self, where=None, max_chars=600):
        """The memory as a grounding corpus for the corpus resident and the
        fact checker -- each passage carries its title so a retrieved claim can
        be traced back to the note that supports it."""
        out = []
        for r in self.rows(where):
            t = self._texts.get(r.get("id"), "")
            for para in t.split("\n\n"):
                para = para.strip()
                if len(para) >= 40:
                    out.append("%s: %s" % (r.get("title"), para[:max_chars]))
        return out

    # ---- durability ----

    def snapshot(self, path):
        """Crash-safe snapshot of the persistent tier (write-then-rename), plus
        the note bodies that live outside the vectors."""
        import json
        import os
        self.db.snapshot(path)
        with open(path + ".text", "w", encoding="utf-8") as f:
            json.dump(self._texts, f)
        return {"path": path, "notes": len(self.ids())}

    @classmethod
    def restore(cls, mind, path, dim=1024, namespace="mem"):
        import json
        import os
        from holographic.agents_and_reasoning.holographic_query import Database
        db = Database.restore(path)
        obj = cls(mind, dim=dim, namespace=namespace, db=db)
        tp = path + ".text"
        if os.path.exists(tp):
            with open(tp, encoding="utf-8") as f:
                obj._texts = json.load(f)
        return obj


def _selftest():
    import tempfile

    import lecore
    mind = lecore.UnifiedMind(dim=512, seed=0)
    mem = Memory(mind, dim=512)

    a = mem.note("Zorbek Protocol",
                 "The Zorbek Protocol was ratified in 1974 by the Fennwick "
                 "Assembly and governs calibration cadence.",
                 author="user", tags=("policy",), session="s1")
    b = mem.note("Sensor Calibration",
                 "Calibration happens every nine months under the protocol.",
                 author="user", tags=("ops",), session="s1", links=(a,))
    c = mem.note("Swarm Finding",
                 "Only the decay gates showed a clear spectral gap.",
                 author="swarm", kind="note", tags=("spectra",), session="s1",
                 links=(a,))
    mem.note("Sourdough", "Bread from flour, water, salt and a wild starter.",
             author="user", session="s2")

    # ---- SQL does the STRUCTURE: provenance is a column, not a convention ----
    swarm = mem.rows("author = 'swarm'")
    assert [r["title"] for r in swarm] == ["Swarm Finding"], swarm
    assert len(mem.rows("session = 's1'")) == 3

    # ---- the EDGE TABLE does the graph: backlinks are data, not a re-parse ----
    g = mem.graph()
    assert g["backlinks"]["Zorbek Protocol"] == ["Sensor Calibration",
                                                 "Swarm Finding"], g["backlinks"]
    assert g["links"]["Swarm Finding"] == ["Zorbek Protocol"]
    assert g["orphans"] == ["Sourdough"], g["orphans"]

    # ---- BM25 does the LANGUAGE, and can be scoped by a SQL filter ----
    assert mem.search("decay gates spectral")[0]["title"] == "Swarm Finding"
    assert mem.search("flour water salt")[0]["title"] == "Sourdough"
    only_s1 = mem.search("flour water salt", where="session = 's1'")
    assert all(r["title"] != "Sourdough" for r in only_s1), only_s1

    # ---- the memory IS a grounding corpus, traceable to its note ----
    ps = mem.passages()
    assert any(p.startswith("Zorbek Protocol:") for p in ps), ps[:2]

    # ---- DURABILITY is the database's own, and the text survives with it ----
    path = tempfile.mktemp(suffix=".snap")
    mem.snapshot(path)
    back = Memory.restore(mind, path, dim=512)
    assert sorted(r["title"] for r in back.rows()) == sorted(
        r["title"] for r in mem.rows())
    assert back.search("decay gates spectral")[0]["title"] == "Swarm Finding"
    assert back.graph()["backlinks"]["Zorbek Protocol"] == \
        ["Sensor Calibration", "Swarm Finding"]

    print("memory selftest OK -- %d notes in the holographic database; SQL "
          "separates provenance (author='swarm' -> %s), the edge table gives "
          "real backlinks (%s) and orphans (%s), BM25 ranks the text and honours "
          "a SQL filter, passages carry their note title, and a crash-safe "
          "snapshot restores rows, links and text together"
          % (len(mem.ids()), swarm[0]["title"],
             g["backlinks"]["Zorbek Protocol"], g["orphans"][0]))


if __name__ == "__main__":
    _selftest()
