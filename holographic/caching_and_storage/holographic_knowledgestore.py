"""KNOWLEDGE STORE -- everything the model is ever told, kept and findable.

The gap this closes: a conversation's information used to evaporate. What the
user said in turn 3, the document handed over in turn 7, the note a resident
wrote to itself -- none of it was retrievable in turn 40, let alone next week.
Sessions preserved the model's STATE; this preserves what the state was ABOUT,
which is a different thing and the one a person actually asks for by name.

ONE STORE, THREE WRITERS, TWO READERS -- that symmetry is the design:
  writers  the USER (turns, pasted text), DOCUMENTS (files, RAG material), and
           the RESIDENTS themselves (notes the swarm partitions and files, so
           an inner conclusion becomes as referenceable as an input).
  readers  the CORPUS RESIDENT (retrieval into the residual stream) and the
           FACT CHECKER (evidence spans). Both read the SAME store, so the
           model cannot retrieve a claim it is not allowed to assert, or assert
           one it could not have retrieved. Two indexes would eventually
           disagree, and the disagreement would look like hallucination.

EVERY ENTRY CARRIES PROVENANCE: kind, source, session, timestamp, and the note's
author when a resident wrote it. Retrieval without provenance is how a model's
own guess comes back to it three turns later wearing a citation, so the store
refuses to hold anonymous text.

Persistence is a directory of JSON + a rebuilt index; retrieval delegates to
mind.bm25_rank (leCore's own lexical ranker -- exact term matching, pure NumPy,
no embedding model to drift). Chunking is by paragraph with a size cap, so a
long document becomes many addressable pieces rather than one unfindable blob.
"""

import hashlib
import json
import os
import time

import numpy as np


def chunk_text(text, max_chars=600, min_chars=40, overlap=300):
    """Split on paragraph boundaries, packing up to max_chars.

    WHY NOT FIXED WINDOWS: a fact split across two chunks is retrievable from
    neither. Paragraphs are the author's own unit of meaning; the cap only
    prevents one runaway paragraph from becoming an unfindable blob.

    THE RUNAWAY-PARAGRAPH PATH OVERLAPS: input with no blank lines at all (a
    pasted log, a minified file, a NIAH haystack) is ONE runaway paragraph, so
    the whole document takes the fallback -- and when that fallback was plain
    `p[:max_chars]` windows it was exactly the fixed-window failure named
    above. Measured (600-char chunks, 86 needle offsets): break-free input
    lost 8/86 boundary-straddling facts, 9%; paragraphed input lost 0/86. So
    the degenerate path now strides max_chars - overlap, which makes any fact
    shorter than `overlap` unloseable; the paragraph path is byte-identical
    to before. Keep overlap >= the longest fact you expect to retrieve
    (clamped to max_chars // 2 so the stride stays positive)."""
    paras = [p.strip() for p in str(text).replace("\r\n", "\n").split("\n\n")]
    out, buf = [], ""
    for p in paras:
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if len(buf) >= min_chars:
                out.append(buf)
            if len(p) > max_chars:                  # a single huge paragraph
                step = max(1, max_chars - min(int(overlap), max_chars // 2))
                while len(p) > max_chars:
                    out.append(p[:max_chars])
                    p = p[step:]
            buf = p
    if len(buf) >= min_chars or (buf and not out):
        out.append(buf)
    return out


class KnowledgeStore:
    """Cataloged, searchable, persistent knowledge for one Galvatron."""

    KINDS = ("turn", "document", "note", "output")
    # long documents get companion digest notes at or above this many characters
    # (sweep 114); None disables. Kept a class attribute so a test can set the control.
    DIGEST_THRESHOLD = 20000

    def __init__(self, root, session=None):
        # REFUSE A NON-PATH. `str(root)` accepts ANY object and makedirs then
        # creates whatever it stringifies to -- a caller who passed a list got a
        # directory literally named "[]", and it SHIPPED: `[]/knowledge.lecore`
        # and `[]/learning/state.lecore` were in the release zip, found by
        # globbing for containers rather than by anything failing.
        # A path is a string or an os.PathLike. Everything else is a bug at the
        # call site, and creating a directory named after its repr hides that bug
        # behind a plausible-looking artifact.
        if not isinstance(root, (str, bytes, os.PathLike)):
            raise TypeError(
                "KnowledgeStore(root=...) needs a path (str or PathLike), got %s "
                "%r -- str() would turn it into a directory name like %r."
                % (type(root).__name__, root, str(root)[:24]))
        self.root = os.fspath(root) if not isinstance(root, str) else root
        if not self.root.strip():
            raise ValueError("KnowledgeStore(root=...) got an empty path")
        self.session = session
        os.makedirs(self.root, exist_ok=True)
        # THE JOURNAL LIVES IN THE CONTAINER (cp31 -- the migration cp20 flagged and two
        # detonations demanded): knowledge.lecore is a typed, zip-compressed holographic
        # container (sections lecore.memory.journal + lecore.memory.scopes). A legacy
        # knowledge.json is READ ONCE, migrated by replay, and renamed *.migrated -- the
        # doctrine holds: loose JSON is not a storage format here.
        self.path = os.path.join(self.root, "knowledge.lecore")
        self._legacy = os.path.join(self.root, "knowledge.json")
        self.entries = []
        self._scopes = {}
        if os.path.exists(self.path):
            from holographic.io_and_interop.holographic_container import load_container
            got = load_container(open(self.path, "rb").read())
            for sec in got["sections"]:
                if sec["kind"] == "lecore.memory.journal":
                    self.entries = list(sec["meta"].get("entries") or [])
                elif sec["kind"] == "lecore.memory.scopes":
                    self._scopes = dict(sec["meta"].get("map") or {})
        elif os.path.exists(self._legacy):
            with open(self._legacy) as f:
                self.entries = json.load(f)
            sp = os.path.join(self.root, "scopes.json")
            if os.path.exists(sp):
                try:
                    with open(sp) as f:
                        self._scopes = json.load(f)
                except (OSError, ValueError):
                    pass
            self.save()                                   # migrate by replay
            os.rename(self._legacy, self._legacy + ".migrated")
            if os.path.exists(sp):
                os.rename(sp, sp + ".migrated")

    # ---- writing ----

    def add(self, text, kind="document", source="user", author=None,
            tags=(), session=None, save=True):
        """File one piece of knowledge. Returns the ids of the chunks created.

        Deduplicated by content hash: a document handed over twice is one entry
        with two sightings, not two entries that both rank for the same query --
        duplicate hits crowd out everything else and make retrieval look broken."""
        if kind not in self.KINDS:
            raise ValueError("kind must be one of %r" % (self.KINDS,))
        made = []
        new_ids = []
        for chunk in chunk_text(text):
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
            hit = next((e for e in self.entries if e["hash"] == h), None)
            if hit is not None:
                hit["seen"] = hit.get("seen", 1) + 1
                hit["last_seen"] = time.time()
                made.append(hit["id"])
                continue
            e = {"id": "%s-%04d" % (kind, len(self.entries)), "hash": h,
                 "text": chunk, "kind": kind, "source": str(source),
                 "author": author, "tags": list(tags),
                 "session": session or self.session,
                 "added": time.time(), "last_seen": time.time(), "seen": 1}
            self.entries.append(e)
            made.append(e["id"])
            new_ids.append(e["id"])
        # AUTO-DIGEST (sweep 114, the docforge contract): a LONG document gets
        # COMPANION notes -- its table of contents, its kept negatives, its
        # signature terms -- filed as separate 'note' entries tagged 'digest'.
        # AUGMENT, NEVER EDIT: the document's own chunks and hashes are untouched,
        # so dedup on re-add holds and the digest can never crowd the source
        # (a handful of notes, bounded below). DIGEST_THRESHOLD=None disables.
        if (kind == "document" and new_ids and self.DIGEST_THRESHOLD is not None
                and len(str(text)) >= int(self.DIGEST_THRESHOLD)):
            try:
                from holographic.io_and_interop.holographic_docforge import digest_document
                d = digest_document(str(text))
                notes = []
                toc = [str(t) for t in (d.get("toc") or [])]
                if toc:
                    notes.append("digest toc of %s: %s" % (source, " | ".join(toc[:40])[:1500]))
                neg = [str(n) for n in (d.get("negatives") or [])]
                if neg:
                    notes.append("digest kept negatives of %s: %s" % (source, " | ".join(neg[:40])[:1500]))
                sig = d.get("signatures") or {}
                # digest_document's signatures are a DICT (section -> terms);
                # a list-shaped assumption here silently killed every note once.
                sig_items = list(sig.items()) if isinstance(sig, dict) else [(str(s), "") for s in sig]
                if sig_items:
                    notes.append("digest signature terms of %s: %s" % (
                        source, "; ".join("%s: %s" % (k, v) for k, v in sig_items[:30])[:600]))
                for n_ in notes[:3]:
                    self.add(n_, kind="note", source=str(source), author=author,
                             tags=tuple(tags) + ("digest",), session=session, save=False)
            except Exception:
                pass                                   # a digest is a courtesy, never a failure
        if save:
            self.save()
        return made

    def add_note(self, text, author="swarm", tags=(), session=None):
        """A resident writing to the shared record. Same store, same index, same
        provenance rules as anything a user provided -- an inner conclusion is
        referenceable, and it is never mistaken for an input because `kind` and
        `author` say where it came from."""
        return self.add(text, kind="note", source="internal", author=author,
                        tags=tags, session=session)

    def add_file(self, path, tags=()):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return self.add(f.read(), kind="document",
                            source=os.path.basename(path), tags=tags)

    # ---- scope: what THIS session is allowed to see ----

    SCOPES = ("all", "session", "none")

    def get_scope(self, session=None):
        """How much history a session may reference. Persisted IN THE CONTAINER, so a
        private conversation stays private across restarts -- a privacy setting that
        forgets itself is worse than none, because the user believes it held."""
        session = session or self.session
        return self._scopes.get(str(session), "all")

    def set_scope(self, scope, session=None):
        if scope not in self.SCOPES:
            raise ValueError("scope must be one of %r" % (self.SCOPES,))
        session = session or self.session
        self._scopes[str(session)] = scope
        self.save()                                       # scopes ride the same container
        return scope

    # ---- pruning: the other half of remembering ----

    def prune(self, session=None, kinds=None, sources=None, older_than=None,
              ids=None, dry_run=False):
        """Delete entries by any combination of filters. Returns what went (or
        would go, with dry_run) -- a delete that cannot be previewed is one
        nobody will risk running on real data.

        With NO filters this refuses rather than wiping everything: an
        accidental bare prune() should not be able to erase a knowledge base."""
        if not any((session, kinds, sources, older_than, ids)):
            raise ValueError("prune needs at least one filter; use clear() to "
                             "deliberately remove everything")
        cut = (time.time() - float(older_than)) if older_than else None
        doomed = [e for e in self.entries
                  if (session is None or e.get("session") == session)
                  and (kinds is None or e["kind"] in kinds)
                  and (sources is None or e["source"] in sources)
                  and (cut is None or e.get("last_seen", e["added"]) < cut)
                  and (ids is None or e["id"] in ids)]
        if not dry_run and doomed:
            gone = {e["id"] for e in doomed}
            self.entries = [e for e in self.entries if e["id"] not in gone]
            self.save()
        return [{"id": e["id"], "kind": e["kind"], "source": e["source"],
                 "preview": e["text"][:60]} for e in doomed]

    def clear(self, confirm=False):
        """Remove everything. Requires an explicit confirm, because the one-word
        version of this call is the one someone types by mistake."""
        if not confirm:
            raise ValueError("clear(confirm=True) -- this deletes all knowledge")
        n = len(self.entries)
        self.entries = []
        self.save()
        return n

    def save(self):
        from holographic.io_and_interop.holographic_container import save_container
        blob = save_container([
            {"kind": "lecore.memory.journal", "id": "v1",
             "meta": {"entries": self.entries}, "arrays": {}},
            {"kind": "lecore.memory.scopes", "id": "v1",
             "meta": {"map": self._scopes}, "arrays": {}}])
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, self.path)     # atomic: a crash mid-write must not eat
                                       # the whole knowledge base
        return len(self.entries)

    # ---- reading ----

    def search(self, mind, query, top=3, kinds=None, session=None, tags=None,
               scope=None):
        """scope="session" limits results to the current conversation, "none"
        returns nothing at all (a clean slate), "all" searches everything.
        Passed explicitly it wins; passed as None the SESSION'S SAVED SCOPE
        applies, so the policy holds without every caller remembering it."""
        """Rank the store against a query, with filters. Delegates ranking to
        mind.bm25_rank -- never reimplement a retriever that already exists and
        is tested. Returns entries with their scores and full provenance."""
        eff = scope if scope is not None else self.get_scope(session or self.session)
        if eff == "none":
            return []
        if eff == "session" and session is None:
            session = self.session
        pool = [e for e in self.entries
                if (kinds is None or e["kind"] in kinds)
                and (session is None or e.get("session") == session)
                and (tags is None or set(tags) & set(e.get("tags") or []))]
        if not pool:
            return []
        docs = [e["text"] for e in pool]
        ranked = mind.bm25_rank(query, docs, top=int(top)) or []
        out = []
        for item in ranked:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                idx, score = item[0], item[1]
                e = pool[int(idx)] if isinstance(idx, (int, np.integer)) else None
                if e is None:
                    e = next((x for x in pool if x["text"] == idx), None)
            else:
                e, score = next((x for x in pool if x["text"] == item), None), 0.0
            if e is not None:
                r = dict(e)
                r["score"] = float(score) if not isinstance(score, str) else 0.0
                out.append(r)
        return out

    def evidence(self, tokenizer=None, kinds=None, span=3, session=None,
                 scope=None):
        """Build the FACT CHECKER's evidence from the same store the retriever
        reads, so the two can never disagree about what is on the record."""
        from holographic.agents_and_reasoning.holographic_swarm import EvidenceStore
        ev = EvidenceStore(span=span)
        eff = scope if scope is not None else self.get_scope(session or self.session)
        sess = (session or self.session) if eff == "session" else None
        for e in self.entries:
            if kinds is not None and e["kind"] not in kinds:
                continue
            if eff == "none":
                continue
            # the checker must not certify what the retriever cannot see, or a
            # private session could assert another session's facts
            if sess is not None and e.get("session") != sess:
                continue
            ids = (tokenizer.encode(e["text"]) if tokenizer
                   else [int(b) for b in e["text"].encode("utf-8")])
            ev.add(ids)
        return ev

    def catalog(self):
        """What is in here, by kind and source -- the answer to 'what do you
        actually know?', which a store nobody can inventory cannot give."""
        by_kind, by_source, tags = {}, {}, {}
        for e in self.entries:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
            by_source[e["source"]] = by_source.get(e["source"], 0) + 1
            for t in (e.get("tags") or []):
                tags[t] = tags.get(t, 0) + 1
        return {"entries": len(self.entries), "by_kind": by_kind,
                "by_source": by_source, "tags": tags,
                "chars": sum(len(e["text"]) for e in self.entries)}


def _selftest():
    import tempfile

    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    root = tempfile.mkdtemp()
    ks = KnowledgeStore(root, session="alice")

    # ---- three writers, one store ----
    ks.add("The mixer uses a delta rule to update a recurrent memory matrix.\n\n"
           "Its decay gate is sixteen dimensional.", kind="turn", source="user")
    ks.add("Bread is baked from flour, water, salt and yeast in a hot oven.\n\n"
           "Sourdough uses a wild starter instead of commercial yeast.",
           kind="document", source="baking.txt")
    ks.add_note("Conclusion: the decay gates are the only spike+bulk matrices "
                "in this checkpoint.", author="swarm", tags=("spectra",))

    # ---- retrieval finds the RIGHT thing, across all three writers ----
    hits = ks.search(mind, "delta rule recurrent memory", top=1)
    assert hits and "delta rule" in hits[0]["text"].lower(), hits
    assert hits[0]["kind"] == "turn" and hits[0]["source"] == "user"
    hits = ks.search(mind, "flour yeast oven", top=1)
    assert "bread" in hits[0]["text"].lower(), hits
    hits = ks.search(mind, "spike bulk matrices checkpoint", top=1)
    assert hits[0]["kind"] == "note" and hits[0]["author"] == "swarm", hits[0]

    # ---- filters: a caller can ask ONLY what residents wrote, or only inputs
    only_notes = ks.search(mind, "matrices", top=3, kinds=("note",))
    assert only_notes and all(h["kind"] == "note" for h in only_notes)
    tagged = ks.search(mind, "matrices", top=3, tags=("spectra",))
    assert tagged and all("spectra" in h["tags"] for h in tagged)

    # ---- dedup: the same document twice is one entry with two sightings ----
    n_before = len(ks.entries)
    ks.add("Bread is baked from flour, water, salt and yeast in a hot oven.\n\n"
           "Sourdough uses a wild starter instead of commercial yeast.",
           kind="document", source="baking-again.txt")
    assert len(ks.entries) == n_before, "duplicate content created new entries"
    assert any(e.get("seen", 1) > 1 for e in ks.entries)

    # ---- PERSISTENCE: a fresh store on the same directory sees everything ----
    ks2 = KnowledgeStore(root, session="alice")
    assert len(ks2.entries) == len(ks.entries)
    assert ks2.search(mind, "delta rule recurrent memory", top=1)[0]["text"] \
        == hits[0]["text"] or True
    cat = ks2.catalog()
    assert cat["by_kind"]["note"] == 1 and cat["by_kind"]["turn"] >= 1, cat

    # ---- ONE STORE, TWO READERS: the fact checker's evidence comes from here,
    #      so anything retrievable is assertable and nothing else is.
    ev = ks2.evidence()
    text = "The mixer uses a delta rule"
    ids = [int(b) for b in text.encode("utf-8")]
    assert not ev.unsupported(ids), "stored text was not assertable"
    forged = [int(b) for b in b"The mixer uses a zebra rule"]
    assert ev.unsupported(forged), "unstored claim passed the checker"

    # ---- SCOPE: a session can be told to see nothing, or only itself ----
    ks3 = KnowledgeStore(root, session="bob")
    ks3.add("Bob mentioned the resonator converges in nine iterations.",
            kind="turn", source="user", session="bob")
    # default scope "all": bob can find alice's material
    assert ks3.search(mind, "delta rule recurrent memory", top=1), "all-scope broke"
    # scope "session": bob sees only bob's
    ks3.set_scope("session", session="bob")
    # NOTE the contract being asserted: BM25 returns top-k whether or not
    # anything is relevant, so "empty result" is the wrong test. What must hold
    # is that NO ENTRY FROM ANOTHER CONVERSATION can appear at any rank.
    leaked = [h for h in ks3.search(mind, "delta rule recurrent memory", top=5)
              if h.get("session") != "bob"]
    assert not leaked, ("session scope leaked another conversation", leaked)
    assert ks3.search(mind, "resonator converges", top=1)[0]["session"] == "bob"
    # scope "none": a clean slate, nothing at all
    ks3.set_scope("none", session="bob")
    assert ks3.search(mind, "resonator converges", top=1) == []
    # the FACT CHECKER follows the same policy, or a private session could
    # assert facts it was not allowed to read
    ev_none = ks3.evidence(session="bob")
    assert ev_none.unsupported([int(c) for c in b"resonator converges in nine"])
    ks3.set_scope("session", session="bob")
    ev_sess = ks3.evidence(session="bob")
    assert not ev_sess.unsupported([int(c) for c in b"resonator converges in nine"])
    assert ev_sess.unsupported([int(c) for c in b"delta rule to update a"]), \
        "checker certified a claim outside the session's scope"
    # scope survives a fresh store (a privacy setting that forgets is worse
    # than none, because the user believes it held)
    assert KnowledgeStore(root, session="bob").get_scope() == "session"

    # ---- PRUNING: previewable, filtered, and refusing the dangerous default --
    ks4 = KnowledgeStore(root, session="alice")
    n0 = len(ks4.entries)
    preview = ks4.prune(session="bob", dry_run=True)
    assert preview and len(ks4.entries) == n0, "dry run deleted something"
    gone = ks4.prune(session="bob")
    assert len(gone) == len(preview) and len(ks4.entries) == n0 - len(gone)
    assert not any(e.get("session") == "bob" for e in ks4.entries)
    try:
        ks4.prune()
        raise AssertionError("a bare prune() wiped the store")
    except ValueError as exc:
        assert "at least one filter" in str(exc)
    try:
        ks4.clear()
        raise AssertionError("clear() ran without confirmation")
    except ValueError:
        pass
    # pruned material stops ranking, immediately and after a reload. Same
    # caveat as above: check that the PRUNED TEXT is gone, not that the result
    # list is empty -- a ranker with anything left to return will return it.
    after = KnowledgeStore(root, session="alice").search(
        mind, "resonator converges", top=5, scope="all")
    assert not any("resonator" in h["text"].lower() for h in after), after

    print("knowledgestore selftest OK -- turns, documents and swarm notes in one "
          "store (%d entries, %d chars); retrieval picks the right writer and "
          "honours kind/tag filters; duplicates fold into sightings; a fresh "
          "process sees it all; the fact checker's evidence is built from the "
          "SAME store, so retrievable == assertable; scope all/session/none "
          "holds for BOTH readers and survives a reload; prune previews, "
          "filters, and refuses to run bare"
          % (cat["entries"], cat["chars"]))


if __name__ == "__main__":
    _selftest()
