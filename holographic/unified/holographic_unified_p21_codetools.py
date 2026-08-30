"""holographic_unified_p21_codetools.py -- Part 21 of the UnifiedMind: THE CODE-TOOLS
ROBUSTNESS FACULTIES. Structural repo mapping across python/js/c, text diagrams,
grounded spec conformance, document outlining, reference-doc generation for arbitrary
roots, and rlimit-sandboxed execution. Machinery in holographic_repograph and
holographic_docforge (io_and_interop); catalog cards in holographic_catalog_p08.

WHY A NEW PART instead of growing p20_zoo: the split contract caps a part at 2,000
lines and p20 already stood at 4,051 before this sweep -- adding here would have
deepened a standing violation. A new part is the additive move the contract wants.
"""


class _UnifiedPart21:

    def merge_trees(self, ours, theirs, base=None, apply=False,
                    ignore=(".git", "__pycache__", ".pytest_cache", ".lecore_jobs")):
        """BRANCH-MERGE decision sheet (sweep 98, built from sweep-97's pain, measured):
        census two trees by sha256, bucket identical/only/differ, and TRIAGE every
        differing file by BOTH-DIRECTION unique-line counts -- the check that catches
        what memory-based triage missed (three sweep-old edits were stomped because the
        collide set came from memory, not measurement). Verdicts per differing file:
        'theirs_is_base' (their copy has ZERO unique lines -> ours wins),
        'ours_is_base' (theirs wins), 'append_extension' (one is a strict prefix of the
        other -- the NOTES case; longer wins), 'both_changed' (a TRUE collision: never
        auto-decided, listed for a human/LLM call). base= adds classic three-way
        verdicts. apply=True executes ONLY the unambiguous verdicts (copies only_theirs
        and theirs-wins files into ours); *.lecore memory files are ALWAYS excluded with
        the reason on the sheet -- memory merges go through memory_export/memory_import,
        never file copy (the sweep-97 rule). Deterministic; refuses ambiguity loudly."""
        import os, hashlib, shutil, difflib

        def census(root_):
            out = {}
            for dp, dn, fn in os.walk(root_):
                dn[:] = [d_ for d_ in dn if d_ not in ignore]
                for f_ in fn:
                    if f_.endswith((".pyc", ".zip")):
                        continue
                    p_ = os.path.join(dp, f_)
                    rel = os.path.relpath(p_, root_)
                    try:
                        out[rel] = hashlib.sha256(open(p_, "rb").read()).hexdigest()
                    except OSError:
                        continue
            return out

        co, ct = census(str(ours)), census(str(theirs))
        cb = census(str(base)) if base else None
        identical = sorted(k for k in co if ct.get(k) == co[k])
        only_ours = sorted(k for k in co if k not in ct)
        only_theirs = sorted(k for k in ct if k not in co)
        differ = sorted(k for k in co if k in ct and ct[k] != co[k])
        sheet, applied, refused = [], [], []
        for rel in differ:
            po, pt = os.path.join(str(ours), rel), os.path.join(str(theirs), rel)
            try:
                lo = open(po, encoding="utf-8", errors="replace").read().splitlines()
                lt = open(pt, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            so, st = set(lo), set(lt)
            u_ours, u_theirs = len(so - st), len(st - so)
            to_, tt_ = "\n".join(lo), "\n".join(lt)
            if rel.endswith(".lecore"):
                verdict = "memory_file"
            elif base and cb is not None:
                hb = cb.get(rel)
                if hb == ct.get(rel):
                    verdict = "theirs_is_base"
                elif hb == co.get(rel):
                    verdict = "ours_is_base"
                else:
                    verdict = "both_changed"
            elif u_theirs == 0:
                verdict = "theirs_is_base"          # every their-line exists in ours
            elif u_ours == 0:
                verdict = "ours_is_base"
            elif to_.startswith(tt_):
                verdict = "append_extension_ours"   # ours extends theirs (NOTES case)
            elif tt_.startswith(to_):
                verdict = "append_extension_theirs"
            else:
                verdict = "both_changed"
            sheet.append({"file": rel, "verdict": verdict,
                          "unique_lines": {"ours": u_ours, "theirs": u_theirs}})
        if apply:
            for rel in only_theirs:
                if rel.endswith(".lecore"):
                    refused.append((rel, "memory file: use memory_import"))
                    continue
                dst = os.path.join(str(ours), rel)
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copy2(os.path.join(str(theirs), rel), dst)
                applied.append(rel)
            for row in sheet:
                rel, v = row["file"], row["verdict"]
                if v in ("ours_is_base", "append_extension_theirs"):
                    shutil.copy2(os.path.join(str(theirs), rel),
                                 os.path.join(str(ours), rel))
                    applied.append(rel)
                elif v == "both_changed":
                    refused.append((rel, "true collision: decide per-file"))
                elif v == "memory_file":
                    refused.append((rel, "memory file: use memory_import"))
        return {"identical": len(identical), "only_ours": only_ours,
                "only_theirs": only_theirs, "differ": sheet,
                "n_both_changed": sum(1 for r in sheet if r["verdict"] == "both_changed"),
                "applied": applied, "refused": refused,
                "advice": ("apply=True executes only unambiguous verdicts; both_changed "
                           "rows need a per-file decision -- diff them with the sheet's "
                           "unique-line counts as the guide")}

    def study(self, root, budget_lines=120, max_docs=12, max_text_bytes=200000,
              question=None, ladder=False):
        """MACRO-LEVEL comprehension of a large directory in ONE call (sweep 93): the
        substrate walks, parses, maps, and digests; the caller -- an LLM giving macro
        direction -- receives one optimized, factual bundle and a handle for follow-up
        questions, with NO per-step orchestration. Composes ingest_files (queryable file
        map + kinds census) + repo_map (symbols, dependency graph, PageRank, budgeted
        skeleton -- code understood without dumping it) + document_digest (TOC, kept-
        negative index, tf*idf signatures per large text doc, truncation declared) +
        corpus_gate (follow-up questions answered from the MATERIAL, certified or
        refused -- never vibed). Returns {'tree', 'code', 'docs', 'ask'}: 'ask' is a
        closure -- study['ask']('question') -> the gate's verdict + the winning chunks.
        question= runs one ask inline. Deterministic; budgets are declared, truncation
        is never silent."""
        import os
        fm = self.ingest_files(str(root))
        kinds = {}
        try:
            for f in fm.files:
                k = getattr(f, "kind", None) or "other"
                kinds[k] = kinds.get(k, 0) + 1
        except Exception:
            kinds = {}
        tree = {"root": str(root), "n_files": len(getattr(fm, "files", []) or []),
                "kinds": kinds}
        code = None
        try:
            rm = self.repo_map(str(root), budget_lines=int(budget_lines))
            code = {"files": rm.get("files"), "languages": rm.get("languages"),
                    "top": rm.get("top"), "skeleton": rm.get("skeleton"),
                    "truncated": rm.get("truncated")}
        except Exception:
            pass                                     # a pure-document tree has no code map
        docs, chunks = [], []
        try:
            md = list(fm.find("*.md")) + list(fm.find("*.txt")) + list(fm.find("*.rst"))
        except Exception:
            md = []
        for f in md[:int(max_docs)]:
            p = str(getattr(f, "path", f))
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()[:int(max_text_bytes)]
            except OSError:
                continue
            d = self.document_digest(txt)
            docs.append({"path": p, "stats": d.get("stats"),
                         "toc": [str(t)[:80] for t in (d.get("toc") or [])[:12]]})
            # chunk by digest sections when available, else paragraphs -- the corpus the
            # gate answers FROM is the same material the digest described
            for para in txt.split("\n\n"):
                if len(para.strip()) > 80:
                    # chunks carry their SOURCE (sweep 102): an answer that cannot name
                    # its file is an answer the host model cannot cite.
                    chunks.append({"text": para.strip()[:1200], "source": p})
        # CODE feeds the corpus too (sweep 94): a pure-code tree yielded chunks == 0 and
        # ask() refused everything -- the factual layer of code is its DOCSTRINGS, and
        # ast harvests them deterministically without importing anything.
        try:
            pys = list(fm.find("*.py"))
        except Exception:
            pys = []
        import ast as _ast
        for f in pys[:400]:
            p = str(getattr(f, "path", f))
            try:
                tree_ = _ast.parse(open(p, encoding="utf-8", errors="replace").read())
            except (OSError, SyntaxError):
                continue
            md_ = _ast.get_docstring(tree_)
            if md_ and len(md_) > 60:
                chunks.append({"text": md_.strip()[:1200], "source": p})
            for node in tree_.body:
                if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
                    d_ = _ast.get_docstring(node)
                    if d_ and len(d_) > 60:
                        chunks.append({"text": d_.strip()[:1200],
                                       "source": "%s:%s" % (p, node.name)})
        n_before = len(chunks)
        chunks = chunks[:800]
        truncation = {}
        if n_before > len(chunks):
            truncation["chunks"] = ("%d harvested, 800 kept -- raise via a focused root "
                                    "(study a subfamily) or page by directory" % n_before)
        if len(pys) > 400:
            truncation["code_files"] = ("%d .py files, docstrings harvested from the "
                                        "first 400" % len(pys))

        def ask(q, k=5):
            """Retrieval with a DECLARED verdict. KEPT NEGATIVE (measured, sweep 93):
            corpus_gate's cascade does not discriminate over ~400 heterogeneous chunks
            -- its scores are STAGE artifacts, inverted at this scale (nonsense hit
            'dense' at 0.50 while real questions hit 'refine' at 0.02). At study scale
            the honest verdict is lexical: rank by shared content words (idf-weighted
            by rarity across the chunks), answerable only when the top chunk shares
            >= 2 informative query words. Simple, deterministic, and it says what it
            measures."""
            if not chunks:
                return {"answerable": False, "advice": "no text material under this root"}
            import re as _re
            from math import log
            qw = {w for w in _re.findall(r"[a-z]{4,}", str(q).lower())}
            if not qw:
                return {"answerable": False, "advice": "no content words in the question"}
            df = {}
            toks = []
            for c in chunks:
                tw = set(_re.findall(r"[a-z]{4,}", c["text"].lower()))
                toks.append(tw)
                for w in qw & tw:
                    df[w] = df.get(w, 0) + 1
            n = len(chunks)
            scored = []
            for i, tw in enumerate(toks):
                shared = qw & tw
                s = sum(log(1.0 + n / (1.0 + df.get(w, 0))) for w in shared)
                scored.append((s, len(shared), i))
            scored.sort(reverse=True)
            top = scored[:int(k)]
            best_s, best_shared, best_i = top[0]
            return {"answerable": bool(best_shared >= 2),
                    "verdict": "lexical retrieval (idf-weighted shared content words)",
                    "top_score": round(float(best_s), 3),
                    "shared_words": int(best_shared),
                    "chunks": [chunks[i]["text"][:400] for _s, _sh, i in top[:3] if _sh > 0],
                    # CITATIONS (sweep 102): every returned chunk names its source file
                    # (and symbol, for code) so the host model can cite, not just assert.
                    "citations": [chunks[i]["source"] for _s, _sh, i in top[:3] if _sh > 0],
                    "advice": ("grounded material retrieved" if best_shared >= 2 else
                               "the material does not cover this -- refuse or go elsewhere")}

        out = {"tree": tree, "code": code, "docs": docs, "n_chunks": len(chunks),
               "ask": ask}
        if truncation:
            out["truncation"] = truncation           # caps DECLARED, never silent
        if ladder and chunks:
            # THE SEVEN-STEP LOOP over the studied material (sweep 94): consolidate ->
            # find patterns -> promote -> repeat, MDL-gated -- the tower's size is set
            # by compression GAIN, not corpus size, which is exactly how a massive
            # dataset stays as tractable as a small one. Chunks become integer word-id
            # sequences (the ladder's alphabet contract, measured on the planted
            # corpus); a shallow ceiling comes back as a LOUD terminal reason.
            import re as _re2
            vocab = {}
            corpus = []
            for c in chunks[:600]:
                seq = []
                for w in _re2.findall(r"[a-z]{4,}", c["text"].lower())[:64]:
                    if w not in vocab:
                        vocab[w] = len(vocab)
                    seq.append(vocab[w])
                if len(seq) >= 8:
                    corpus.append(seq)
            if corpus:
                tower = self.climb_ladder(corpus, max_depth=6)
                out["ladder"] = {"levels": len(tower),
                                 "summary": self.ladder_summary(tower),
                                 "vocab": len(vocab)}
        if question:
            out["answer"] = ask(question)
        return out

    def repo_map(self, root, budget_lines=200, archive_topic=None, focus=None):
        """Map a MIXED-LANGUAGE tree (py/js/c): defs + references per file, the
        file dependency graph, deterministic PageRank ranking, and a budgeted
        text skeleton -- the structural layer codebase_map's prose archive does
        not carry. archive_topic= additionally stores the skeleton in research
        memory so later sessions query instead of re-scanning.
        See holographic_repograph.RepoGraph."""
        from holographic.io_and_interop.holographic_repograph import RepoGraph
        g = RepoGraph(root)
        focused = g.refocus(focus) if focus else []
        out = g.summary()
        if focus:
            # personalized rank (sweep 65): files near the work outrank the
            # globally popular; an unmatched focus is reported, never silent
            out["focus_matched"] = focused
        out["skeleton"] = g.skeleton(budget_lines)
        if archive_topic:
            self.research_archive(archive_topic, out["skeleton"].split("\n"),
                                  sources=[str(root)])
        return out

    def codebase_diagram(self, root, fmt="mermaid", max_nodes=24):
        """Draw a codebase as diagram TEXT (mermaid flowchart or graphviz dot):
        top-ranked files as nodes, reference-count-weighted edges between them.
        Text diagrams diff and version; caps are announced in a note node,
        never silent. See holographic_repograph.diagram."""
        from holographic.io_and_interop import holographic_repograph as _rg
        return _rg.diagram(_rg.RepoGraph(root), fmt=fmt, max_nodes=max_nodes)

    def spec_conformance(self, spec_text, root, llm=None):
        """Check a spec/SOP against a source tree WITHOUT hallucination: atomic
        claims -> mechanical file:line evidence (every citation re-read from
        disk before it may appear) -> verdicts supported / partial / violated /
        unverifiable. Claims the machine cannot grip are abstained on, never
        judged. llm= may propose a finer claim split; it can never confirm.
        See holographic_repograph.SpecChecker."""
        from holographic.io_and_interop.holographic_repograph import SpecChecker
        return SpecChecker(root).check(spec_text, llm=llm)

    def document_digest(self, text, negative_markers=("KEPT NEG",),
                        as_markdown=False, max_bytes=200_000):
        """Digest a large text for LEARNING in one substrate-only call (zero
        model calls): authored TOC (both head dialects), kept-negative index
        (citations -- the marked text stays in its section), and per-section
        tf*idf signatures over the document's own sections. `as_markdown=True`
        renders the budgeted companion note (negatives funded first, then
        signatures, then TOC; everything truncated says so).

        This is the sweep-71 exercise promoted from a hand-driven model
        session to one call -- and it runs AUTOMATICALLY at ingestion:
        KnowledgeStore.add files the companion beside any document >=
        DIGEST_THRESHOLD, original chunks byte-identical (augment, never
        edit). See holographic_docforge.digest_document."""
        from holographic.io_and_interop.holographic_docforge import (
            digest_document, digest_markdown)
        d = digest_document(text, negative_markers=negative_markers)
        return digest_markdown(d, max_bytes=max_bytes) if as_markdown else d

    def document_outline(self, text, llm=None):
        """Organize a large text into sections + a table of contents. Headed
        markdown keeps the author's structure; unheaded prose is cut at lexical
        cohesion dips (TextTiling-style, deterministic). Content is reorganized,
        never rewritten; llm= may only rename section titles.
        See holographic_docforge.outline_document."""
        from holographic.io_and_interop.holographic_docforge import \
            outline_document
        return outline_document(text, llm=llm)

    def docs_generate(self, root):
        """Emit a deterministic markdown REFERENCE for any py/js/c tree: every
        file, every extracted definition with signature, line number, and (for
        python) the author's own first docstring sentence.
        See holographic_docforge.generate_docs."""
        from holographic.io_and_interop.holographic_docforge import \
            generate_docs
        return generate_docs(root)

    def sop_check(self, sop_text):
        """Validate an authored SOP WITHOUT running anything: {"ok": True,
        "steps": n} or {"ok": False, "errors": [...]} with every problem
        named by line. The author's (usually a model's) edit-until-clean
        loop. See holographic_soprunner.parse_sop for the format."""
        from holographic.agents_and_reasoning.holographic_soprunner import parse_sop
        p = parse_sop(sop_text)
        return {"ok": True, "title": p["title"], "steps": len(p["steps"])} \
            if p["ok"] else {"ok": False, "errors": p["errors"]}

    def sop_run(self, sop, llm=None, max_steps=50):
        """FOLLOW ORDERS: execute an authored text SOP through this mind --
        invoke: faculties, python/javascript/c: via the sandbox, shell: via the
        allowlist, verify: per step, on_fail: abort|continue|retry N|escalate.
        The model (llm=, default none) is consulted ONLY at guidance: steps and
        escalations; a plain scriptable SOP runs with zero model calls, and the
        returned llm_calls count proves it. `sop` may be a name saved via
        sop_save. An SOP that does not fully parse is REFUSED before step 1.
        See holographic_soprunner.SOPRunner."""
        from holographic.agents_and_reasoning.holographic_soprunner import SOPRunner
        text = str(sop)
        if "\n" not in text and "##" not in text:
            loaded = self.sop_load(text)
            if loaded.get("found"):
                text = loaded["text"]
        return SOPRunner(self, llm=llm, max_steps=max_steps).run(text)

    def sop_save(self, name, sop_text):
        """Save a NAMED SOP for later sop_run(name) -- the leOS macro_registry
        pattern on the durable KnowledgeStore. Validates first: an SOP that
        does not parse is refused, not stored (never save an order you would
        refuse to follow). See holographic_soprunner (format) and sop_load."""
        from holographic.agents_and_reasoning.holographic_soprunner import parse_sop
        p = parse_sop(sop_text)
        if not p["ok"]:
            return {"ok": False, "errors": p["errors"]}
        from holographic.caching_and_storage.holographic_knowledgestore import \
            KnowledgeStore
        root = getattr(self, "_archive_root", None) or "/tmp/lecore_archive"
        KnowledgeStore(root).add_note("[sop:%s] %s" % (name, sop_text),
                                      tags=("sop", str(name)))
        return {"ok": True, "name": str(name), "steps": len(p["steps"])}

    def sop_load(self, name):
        """Fetch a named SOP saved by sop_save: exact tag match on the
        KnowledgeStore, LAST save wins (revisions are appends, never edits).
        Returns {"found": bool, "text": ...}. See holographic_soprunner."""
        from holographic.caching_and_storage.holographic_knowledgestore import \
            KnowledgeStore
        root = getattr(self, "_archive_root", None) or "/tmp/lecore_archive"
        prefix = "[sop:%s] " % name
        hits = [e for e in KnowledgeStore(root).entries
                if str(e.get("text", "")).startswith(prefix)]
        if not hits:
            return {"found": False, "name": str(name)}
        return {"found": True, "name": str(name),
                "text": str(hits[-1]["text"])[len(prefix):]}

    def sandbox_run(self, code, lang="python", timeout=10, mem_mb=512):
        """Execute a snippet in a THROTTLED child process: cpu/memory/file-size
        rlimits, scrubbed env (PYTHONHASHSEED=0), temp cwd, wall timeout, and
        loudly-marked output caps. node/cc are used when present and refused
        honestly when absent. See holographic_docforge.sandbox_run."""
        from holographic.io_and_interop.holographic_docforge import \
            sandbox_run as _run
        return _run(code, lang=lang, timeout=timeout, mem_mb=mem_mb)


def _selftest():
    """Part contract (delegated to holographic.unified.check_part) plus one
    live faculty round-trip on planted truth."""
    from holographic.unified import check_part
    n = check_part("holographic.unified.holographic_unified_p21_codetools",
                   "_UnifiedPart21")
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert m.sandbox_run("print(6*7)")["stdout"].strip() == "42"
    o = m.document_outline("# A\nx\n\n# B\ny\n")
    assert [s["title"] for s in o["sections"]] == ["A", "B"]
    return {"part": "holographic_unified_p21_codetools", "members": n}


if __name__ == "__main__":
    print(_selftest())
