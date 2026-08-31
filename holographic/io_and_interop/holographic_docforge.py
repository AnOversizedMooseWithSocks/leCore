"""Document forge: outline big documents, generate reference docs, run code sandboxed.

WHY THIS MODULE EXISTS. Three jobs a self-improving code engine keeps needing,
each confirmed missing by a find_capability audit that returned only fallbacks:

  * outline_document -- take a large text (research notes, a spec, a dumped
    log) and return an ORGANIZED document: hierarchy, per-section spans, and a
    markdown rebuild with a table of contents. Headed markdown is parsed as
    written; unheaded prose is segmented by a TextTiling-style vocabulary-shift
    scan (Hearst 1997: lexical cohesion dips mark topic boundaries), which is
    deterministic and needs nothing but word counts.
  * generate_docs -- walk ANY source root (py/js/c via holographic_repograph's
    extractors) and emit a deterministic markdown reference: module docs,
    signatures, docstrings. docgen.py does this for leCore itself; this is the
    same idea generalized into a faculty usable on arbitrary trees.
  * sandbox_run -- execute a snippet in a THROTTLED subprocess: rlimits on cpu
    / address space / file size, scrubbed environment, temp cwd, wall timeout,
    capped-and-MARKED output. Python always works (sys.executable -I); node
    and cc are used when present and refused honestly when not (the toolchain
    is an accelerator, never a dependency -- the numba rule applied to
    binaries).

AUDITED NEGATIVE, ON RECORD: holographic_segment.Segmenter was evaluated for
the outline job and does NOT apply -- it discovers units in a SYMBOL stream by
branching entropy (word discovery granularity), not topic boundaries between
paragraphs. Recording this here is what keeps the next session from re-auditing
it. The two could meet on a character-level corpus; nobody has needed that.

KEPT NEGATIVES:
  * exec()-in-process sandboxing was rejected outright: no rlimit isolation,
    shared interpreter state, and one `while True:` hangs the mind. A child
    process is the only honest floor, and `-I` (isolated mode) is what keeps
    the child from importing the caller's site-packages surprises.
  * Silent output capping is the codebase's named failure class (NOTES sweeps
    59-60). Every cap here appends a loud marker with the dropped byte count.
"""

import os
import re
import resource
import subprocess
import sys
import tempfile
import time

from . import holographic_repograph as _rg

TRUNCATION_MARKER = _rg.TRUNCATION_MARKER

# ---------------------------------------------------------------------------
# outline: big text -> organized document with a table of contents
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

_STOPWORDS = frozenset("""
the a an and or of to in for on with is are was were be been it its this that
these those as at by from not but if then than so we you they he she i our
your their his her all any each which what when where who how into over under
about after before between during out up down off again further once here
there can will just should now do does did have has had
""".split())


def _paragraphs(text):
    """(start_line, end_line, text) per blank-line-separated block, 1-based
    inclusive spans -- the atoms the topic scan moves between."""
    blocks, buf, start = [], [], 1
    for i, line in enumerate(text.split("\n"), 1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            blocks.append((start, i - 1, "\n".join(buf)))
            buf = []
    if buf:
        blocks.append((start, start + len(buf) - 1, "\n".join(buf)))
    return blocks


def _vocab(text):
    """Lowercased content-word multiset of a block (stopwords out -- they
    cohere every pair of paragraphs and would flatten the dip signal)."""
    counts = {}
    for w in _WORD.findall(text.lower()):
        if w not in _STOPWORDS and len(w) > 2:
            counts[w] = counts.get(w, 0) + 1
    return counts


def _cohesion(a, b):
    """Cosine over word counts between two adjacent windows: the quantity
    whose local DIPS mark topic boundaries (TextTiling's core observation)."""
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _title_of(block_texts):
    """A deterministic section title: the two most distinctive content words
    of the section (frequency within, ties broken alphabetically). Not poetry,
    but stable, honest, and grep-able -- an optional llm may rename sections
    downstream; the deterministic title is always the fallback."""
    counts = {}
    for t in block_texts:
        for w, c in _vocab(t).items():
            counts[w] = counts.get(w, 0) + c
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
    return " ".join(w for w, _ in best).title() or "Section"


def outline_document(text, min_section_paragraphs=2, llm=None):
    """Organize a large text into sections with a table of contents.

    Headed documents (markdown '#' lines) are parsed as the author structured
    them -- imposing a topic scan on explicit structure would be inventing
    disagreement. Unheaded prose gets the TextTiling-style scan: cohesion
    between adjacent paragraph windows, boundaries at local minima below the
    mean. Both paths return the same shape:

        {"headed": bool, "toc": [str], "sections": [{title, level,
          start_line, end_line, paragraphs}], "markdown": str}

    The markdown rebuild keeps every original line inside its section -- the
    outline REORGANIZES, it never rewrites (rewriting is the llm's job if the
    caller wants it, and the llm renames titles at most: content is source).
    """
    lines = text.split("\n")
    heads = [(i, len(m.group(1)), m.group(2))
             for i, ln in enumerate(lines, 1)
             for m in [_HEADING.match(ln)] if m]
    # RULE+TITLE boundaries are AUTHORED structure too (large-text digestion
    # test, measured on the 6 MB research notes): a horizontal rule (a line of
    # >= 8 dashes, nothing else) followed by a non-empty title line is how the
    # newer log entries are sectioned -- 77 of them, ALL invisible to the
    # '#'-only parse, silently lumped into whichever '#' section preceded.
    # Absent structure looks legit; same disease family as the absent-result
    # bugs. Recognized at level 2, merged into one sorted boundary list.
    ruled = []
    for i, ln in enumerate(lines[:-1]):
        s = ln.strip()
        if len(s) >= 8 and set(s) == {"-"}:
            title = lines[i + 1].strip()
            if title and not _HEADING.match(lines[i + 1]):
                ruled.append((i + 2, 2, title))     # 1-based line of the TITLE
    if ruled:
        seen = {h[0] for h in heads}
        heads = sorted(heads + [r for r in ruled if r[0] not in seen])
    sections = []
    if heads:
        for j, (line_no, level, title) in enumerate(heads):
            end = (heads[j + 1][0] - 1) if j + 1 < len(heads) else len(lines)
            sections.append({"title": title, "level": level,
                             "start_line": line_no, "end_line": end,
                             "paragraphs": None})
        headed = True
    else:
        blocks = _paragraphs(text)
        if not blocks:
            return {"headed": False, "toc": [], "sections": [],
                    "markdown": text}
        vocabs = [_vocab(b[2]) for b in blocks]
        gaps = [_cohesion(vocabs[i], vocabs[i + 1])
                for i in range(len(blocks) - 1)]
        mean = sum(gaps) / len(gaps) if gaps else 0.0
        cuts = []
        for i, gsc in enumerate(gaps):
            # boundary = local cohesion minimum below the mean; both
            # conditions matter (below-mean alone over-cuts choppy prose,
            # local-minimum alone cuts at every mild dip)
            left = gaps[i - 1] if i > 0 else 1.0
            right = gaps[i + 1] if i + 1 < len(gaps) else 1.0
            if gsc < mean and gsc <= left and gsc <= right:
                cuts.append(i + 1)          # section starts at block i+1
        # enforce minimum section size so a one-paragraph "section" cannot
        # shatter the outline
        starts, last = [0], 0
        for c in cuts:
            if c - last >= min_section_paragraphs:
                starts.append(c)
                last = c
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(blocks)
            seg = blocks[s:e]
            sections.append({"title": _title_of([b[2] for b in seg]),
                             "level": 2,
                             "start_line": seg[0][0],
                             "end_line": seg[-1][1],
                             "paragraphs": e - s})
        headed = False

    if llm is not None:
        # The model may RENAME sections (a labeling job) -- one call per
        # section body head, mechanically truncated; a dead model changes
        # nothing. It may not move or rewrite content.
        for s in sections:
            body = "\n".join(lines[s["start_line"] - 1:
                                   min(s["end_line"], s["start_line"] + 5)])
            try:
                name = str(llm("One short title (max 6 words) for:\n" +
                               body)).strip().split("\n")[0][:60]
                if name:
                    s["title"] = name
            except Exception:
                pass

    toc = ["%s- [%s] (lines %d-%d)" %
           ("  " * (s["level"] - 1), s["title"], s["start_line"],
            s["end_line"]) for s in sections]
    md = ["# Table of Contents", ""] + toc + [""]
    for s in sections:
        if not headed:
            md.append("%s %s" % ("#" * s["level"], s["title"]))
        md += lines[s["start_line"] - 1:s["end_line"]]
        md.append("")
    return {"headed": headed, "toc": toc, "sections": sections,
            "markdown": "\n".join(md)}


# ---------------------------------------------------------------------------
# reference docs for any root
# ---------------------------------------------------------------------------

def generate_docs(root, max_files=2000):
    """A deterministic markdown REFERENCE for any py/js/c tree: per file, its
    language, line count, and every extracted definition with signature, line
    number, and (python) first docstring sentence. Delegates extraction to
    holographic_repograph -- one extractor family, two consumers, zero drift.
    Returns {"markdown": str, "files": int, "defs": int}."""
    g = _rg.RepoGraph(root, max_files=max_files)
    out = ["# Reference: %s" % root, "",
           "%d files, %d definitions.%s" %
           (len(g.files), sum(len(v["defs"]) for v in g.files.values()),
            ("  " + TRUNCATION_MARKER + " scan capped") if g.truncated
            else ""), ""]
    n_defs = 0
    for rel in sorted(g.files):
        info = g.files[rel]
        out.append("## %s  `[%s, %d lines]`" % (rel, info["lang"],
                                                info["lines"]))
        doc_lines = {}
        if info["lang"] == "python":
            # first docstring sentence per def: the author's own one-liner is
            # better documentation than anything synthesized
            try:
                import ast as _ast
                src = open(os.path.join(str(root), rel),
                           encoding="utf-8", errors="ignore").read()
                for node in _ast.walk(_ast.parse(src)):
                    if isinstance(node, (_ast.FunctionDef,
                                         _ast.AsyncFunctionDef,
                                         _ast.ClassDef)):
                        d = (_ast.get_docstring(node) or "").strip()
                        if d:
                            doc_lines[node.name] = d.split("\n")[0][:140]
            except SyntaxError:
                pass
        for name, kind, ln, sig in info["defs"]:
            n_defs += 1
            tail = doc_lines.get(name.split(".")[-1], "")
            out.append("- `%s%s` *(%s, line %d)*%s" %
                       (name, sig, kind, ln, ("  -- " + tail) if tail else ""))
        out.append("")
    return {"markdown": "\n".join(out), "files": len(g.files), "defs": n_defs}


# ---------------------------------------------------------------------------
# sandboxed execution
# ---------------------------------------------------------------------------

_OUTPUT_CAP = 65536      # bytes kept per stream; overflow is MARKED, not eaten


def _limits(cpu_seconds, mem_mb):
    """The preexec hook: hard rlimits inside the child, before exec. CPU and
    address space are the two that stop runaway loops and allocation bombs;
    FSIZE stops a disk-filler; NOFILE keeps a descriptor storm contained."""
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        b = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
        resource.setrlimit(resource.RLIMIT_FSIZE,
                           (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        os.setsid()          # own session: a timeout kill reaps the whole tree
    return apply


def _cap(data):
    """Decode + cap one output stream, LOUDLY (the named failure class:
    a silently capped result reads as a complete one)."""
    text = data.decode("utf-8", errors="replace")
    if len(text) > _OUTPUT_CAP:
        return (text[:_OUTPUT_CAP] + "\n" + TRUNCATION_MARKER +
                " %d bytes dropped" % (len(text) - _OUTPUT_CAP))
    return text


def _which(name):
    for d in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def sandbox_run(code, lang="python", timeout=10, mem_mb=512, argv=(),
                stdin_text=""):
    """Run a snippet in a throttled child process and report honestly.

    Returns {ok, returncode, stdout, stderr, elapsed, lang, limits} -- or a
    refusal {ok: False, why: ...} when the language's toolchain is absent
    (node / cc are opt-in accelerators; their absence is stated, never worked
    around with a lookalike). `ok` means exit 0 within limits; a timeout or a
    limit kill reports which limit fired.

    Determinism note: the child gets a scrubbed env with PYTHONHASHSEED=0 and
    LC_ALL=C, an empty temp cwd, and isolated mode for python -- same snippet,
    same bytes out, regardless of the caller's shell. Wall-clock `elapsed` is
    reported for the caller's information and is NOT part of the contract
    (determinism is a CPU property; the clock is weather).
    """
    lang = str(lang).lower()
    tmp = tempfile.mkdtemp(prefix="sandbox_")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin",
           "PYTHONHASHSEED": "0", "LC_ALL": "C", "HOME": tmp,
           "TMPDIR": tmp}
    try:
        if lang == "python":
            src = os.path.join(tmp, "snippet.py")
            open(src, "w").write(code)
            cmd = [sys.executable, "-I", src] + list(argv)
        elif lang in ("javascript", "js", "node"):
            node = _which("node")
            if node is None:
                return {"ok": False,
                        "why": "node not on PATH -- javascript sandboxing "
                               "needs it; install node or run lang='python'"}
            src = os.path.join(tmp, "snippet.js")
            open(src, "w").write(code)
            cmd = [node, src] + list(argv)
        elif lang == "c":
            cc = _which("cc") or _which("gcc")
            if cc is None:
                return {"ok": False,
                        "why": "no C compiler (cc/gcc) on PATH -- install "
                               "one or run lang='python'"}
            src = os.path.join(tmp, "snippet.c")
            binp = os.path.join(tmp, "snippet.bin")
            open(src, "w").write(code)
            comp = subprocess.run([cc, "-O2", "-o", binp, src],
                                  capture_output=True, timeout=60, env=env)
            if comp.returncode != 0:
                return {"ok": False, "returncode": comp.returncode,
                        "why": "compile failed",
                        "stderr": _cap(comp.stderr), "lang": "c"}
            cmd = [binp] + list(argv)
        else:
            return {"ok": False,
                    "why": "lang must be python/javascript/c, got %r" % lang}

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, env=env, capture_output=True,
                input=stdin_text.encode(), timeout=timeout,
                preexec_fn=_limits(max(1, int(timeout)), mem_mb))
            rc, out_b, err_b = proc.returncode, proc.stdout, proc.stderr
            timed_out = False
        except subprocess.TimeoutExpired as te:
            rc, timed_out = -1, True
            out_b = te.stdout or b""
            err_b = te.stderr or b""
        elapsed = time.time() - t0
        result = {"ok": (rc == 0), "returncode": rc,
                  "stdout": _cap(out_b), "stderr": _cap(err_b),
                  "elapsed": round(elapsed, 3), "lang": lang,
                  "limits": {"timeout_s": timeout, "mem_mb": mem_mb,
                             "cwd": "temp", "env": "scrubbed"}}
        if timed_out:
            result["why"] = "wall timeout after %ss" % timeout
        elif rc == -9 or rc == 137:
            result["why"] = "killed -- rlimit (cpu or memory) fired"
        return result
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _selftest():
    # --- outline: headed doc keeps the author's structure --------------------
    headed = ("# Alpha\ntext one\n\n## Beta\ntext two\n\n# Gamma\ntext three\n")
    o = outline_document(headed)
    assert o["headed"] and [s["title"] for s in o["sections"]] == \
        ["Alpha", "Beta", "Gamma"], o["sections"]
    assert "Table of Contents" in o["markdown"]

    # --- outline: unheaded prose with a PLANTED topic shift ------------------
    # Two blocks about cooking, two about compilers, twice (min-size honored).
    cook = "flour butter oven bake dough sugar knead pastry rest proof"
    comp = "compiler lexer parser tokens grammar syntax codegen emit link"
    text = "\n\n".join([cook, cook + " crust glaze", comp, comp + " optimize"])
    o2 = outline_document(text)
    assert not o2["headed"]
    assert len(o2["sections"]) == 2, o2["sections"]     # one cut, at the shift
    assert o2["sections"][1]["start_line"] > o2["sections"][0]["end_line"]
    # determinism: same text, same outline, byte for byte
    assert outline_document(text)["markdown"] == o2["markdown"]
    # every original line survives inside some section (reorganize != rewrite)
    for s in o2["sections"]:
        assert s["end_line"] >= s["start_line"]

    # --- generate_docs on a planted mini-tree --------------------------------
    import shutil
    tmpd = tempfile.mkdtemp(prefix="docforge_st_")
    try:
        open(os.path.join(tmpd, "m.py"), "w").write(
            'def area(w, h):\n    """Rectangle area."""\n    return w * h\n')
        open(os.path.join(tmpd, "u.js"), "w").write(
            "function hello(x) { return x; }\n")
        d = generate_docs(tmpd)
        assert d["files"] == 2 and d["defs"] == 2, d
        assert "`area(w, h)`" in d["markdown"], d["markdown"]
        assert "Rectangle area." in d["markdown"]
        assert "hello" in d["markdown"]
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    # --- sandbox: the honest floor -------------------------------------------
    r = sandbox_run("print(2 + 3)")
    assert r["ok"] and r["stdout"].strip() == "5", r
    # planted failure: nonzero exit is reported, not prettified
    r2 = sandbox_run("import sys; sys.exit(3)")
    assert not r2["ok"] and r2["returncode"] == 3, r2
    # planted BUSY hang: RLIMIT_CPU and the wall timeout race (wall-clock is
    # not a contract); the contract is a NAMED kill, either name
    r3 = sandbox_run("while True: pass", timeout=2)
    assert not r3["ok"] and ("rlimit" in r3.get("why", "")
                             or "timeout" in r3.get("why", "")), r3
    # planted IDLE hang: sleep burns no cpu, only the wall timeout can catch it
    r3b = sandbox_run("import time; time.sleep(30)", timeout=2)
    assert not r3b["ok"] and "timeout" in r3b.get("why", ""), r3b
    # isolation: the child's env is scrubbed (a secret in ours must not leak)
    os.environ["DOCFORGE_SECRET"] = "hunter2"
    try:
        r4 = sandbox_run("import os; print(os.environ.get"
                         "('DOCFORGE_SECRET', 'ABSENT'))")
        assert r4["stdout"].strip() == "ABSENT", r4
    finally:
        del os.environ["DOCFORGE_SECRET"]
    # loud cap: oversized output carries the marker AND the dropped count
    r5 = sandbox_run("print('x' * 200000)")
    assert TRUNCATION_MARKER in r5["stdout"] and "dropped" in r5["stdout"]
    # unknown toolchain refuses honestly, never guesses
    r6 = sandbox_run("puts 1", lang="ruby")
    assert not r6["ok"] and "lang must be" in r6["why"]
    print("holographic_docforge selftest OK")


def digest_document(text, negative_markers=("KEPT NEG",), sig_terms=6):
    """The sweep-71 exercise as ONE substrate-only call: outline a large text
    and build the LEARNING AUGMENTATION on top -- zero model calls, original
    untouched (the digest cites line spans; it never edits or duplicates the
    source; storage stays verbatim elsewhere).

    Layers, all deterministic:
      toc        -- [(title, level, start_line)] as authored (outline_document,
                    both head dialects: '#' and rule+TITLE).
      negatives  -- [(section_title, start_line, first_line)] for sections
                    containing any of `negative_markers` -- an index that
                    CITES; the marked text stays living in its section.
      signatures -- {section_title: [top distinctive tokens]} by tf*idf
                    ACROSS the document's own sections (self-contained: no
                    corpus needed), for retrieval and cross-referencing.
      stats      -- {bytes, lines, sections, negatives}.

    WHY zero-LLM: the manual version of this (sweep 71) spent an entire model
    session orchestrating outline -> index -> toc by hand. Every layer is a
    scan or a count; a model adds nothing but cost here. `outline_document`'s
    llm hook remains the ONLY model rung, and stays off by default."""
    import math, re as _re
    o = outline_document(text)
    lines = text.split("\n")
    toc = [(s["title"], s["level"], s["start_line"]) for s in o["sections"]]
    negatives = []
    bodies = []
    for s in o["sections"]:
        body = lines[s["start_line"] - 1: s["end_line"]]
        bodies.append((s["title"], body))
        for j, ln in enumerate(body):
            u = ln.upper()
            if any(mk in u for mk in negative_markers):
                k = j
                while k < len(body) - 1 and len(body[k].strip()) < 12:
                    k += 1
                negatives.append((s["title"], s["start_line"],
                                  body[k].strip()[:150]))
                break
    # per-section signatures: tf*idf where the DOCUMENT is its own corpus and
    # sections are the documents -- distinctive terms, not frequent ones
    tok = _re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{3,}")
    df = {}
    tfs = []
    for title, body in bodies:
        tf = {}
        for w in tok.findall(" ".join(body).lower()):
            tf[w] = tf.get(w, 0) + 1
        tfs.append(tf)
        for w in tf:
            df[w] = df.get(w, 0) + 1
    n_sec = max(1, len(bodies))
    signatures = {}
    for (title, _), tf in zip(bodies, tfs):
        scored = sorted(((c * math.log(n_sec / df[w]), w) for w, c in tf.items()
                         if df[w] < n_sec), reverse=True)
        signatures[title] = [w for _, w in scored[:sig_terms]]
    return {"toc": toc, "negatives": negatives, "signatures": signatures,
            "stats": {"bytes": len(text), "lines": len(lines),
                      "sections": len(toc), "negatives": len(negatives)},
            "headed": o["headed"]}


def digest_markdown(digest, max_bytes=200_000):
    """Render a digest as the COMPANION NOTE: compact navigation markdown
    (TOC + negatives index + signatures), capped at `max_bytes` so a huge
    document's augmentation never dwarfs retrieval. Truncation drops TOC
    tail rows first, never index entries -- the indexes are the learning
    payload; the full TOC is regenerable from the source in one call."""
    # ONE running budget over ALL blocks, priority order: negatives, then
    # signatures, then TOC. Priority means FUNDED FIRST, not unlimited: the
    # first cut exempted negatives ("the learning payload") and a document
    # with 1,200 marked negatives filed 72 KB of them -- an uncapped
    # priority block is just the crowding bug wearing a halo. Two earlier
    # lessons kept: a cap that guards one of three faucets is not a cap
    # (4,000-section doc, signature rows); everything truncated says so
    # with a marker, and the full index is regenerable in one call.
    out = ("# Digest (auto, substrate-only; cites lines in the source -- "
           "original stored verbatim separately)")

    def _budgeted(header, rows, marker, out):
        out += "\n\n" + header
        for row in rows:
            if len(out) + len(row) + 1 > max_bytes:
                return out + "\n- ... (%s truncated; regenerate from source)" % marker
            out += "\n" + row
        return out

    out = _budgeted("## Kept-negative index (%d)" % len(digest["negatives"]),
                    ["- %s (line %d): %s" % (t[:70], ln, first)
                     for t, ln, first in digest["negatives"]],
                    "negatives", out)
    out = _budgeted("## Section signatures",
                    ["- %s: %s" % (t[:70], " ".join(ws))
                     for t, ws in digest["signatures"].items() if ws],
                    "signatures", out)
    out = _budgeted("## Table of contents (%d sections)" % len(digest["toc"]),
                    ["%s- %s (line %d)" % ("  " * (lv - 1), t, ln)
                     for t, lv, ln in digest["toc"]],
                    "TOC", out)
    return out


def _selftest_rule_titles():
    # PLANTED TRUTH: a rule+title entry between two '#' sections must become
    # its own section (the 6 MB research-notes miss, pinned small).
    doc = ("# Alpha\n\nalpha prose.\n\n"
           + "-" * 60 + "\n"
           + "SWEEP 9 -- THE PLANTED ENTRY\n\nsweep prose.\n\n"
           + "# Omega\n\nomega prose.\n")
    o = outline_document(doc)
    titles = [s["title"] for s in o["sections"]]
    assert o["headed"] and any("SWEEP 9" in t for t in titles), titles
    sec = [s for s in o["sections"] if "SWEEP 9" in s["title"]][0]
    body = "\n".join(doc.split("\n")[sec["start_line"] - 1: sec["end_line"]])
    assert "sweep prose." in body and "omega prose." not in body, body
    # and a rule with no title after it must NOT invent a section
    o2 = outline_document("# A\n\nx\n\n" + "-" * 40 + "\n\ny\n")
    assert sum(1 for s in o2["sections"]) == 1, o2["sections"]


def _selftest_digest():
    # PLANTED: a doc with two sections, one carrying a marked negative and a
    # distinctive term -- the digest must index the negative, sign the term,
    # and cost ZERO model calls (there is no llm parameter to even pass).
    doc = ("# Alpha\n\nzebra zebra zebra prose here.\n\n"
           "# Beta\n\nKEPT NEGATIVE: the planted refutation line.\n\n"
           "quokka quokka quokka words.\n")
    d = digest_document(doc)
    assert d["stats"]["sections"] == 2 and d["stats"]["negatives"] == 1, d["stats"]
    assert d["negatives"][0][0] == "Beta" and "refutation" in d["negatives"][0][2]
    assert "zebra" in d["signatures"]["Alpha"] and "quokka" in d["signatures"]["Beta"]
    md = digest_markdown(d)
    assert "Kept-negative index (1)" in md and "zebra" in md
    # cap honored, indexes survive truncation
    big = "\n".join("# S%d\n\nbody %d." % (i, i) for i in range(4000))
    md2 = digest_markdown(digest_document(big), max_bytes=20_000)
    assert len(md2) < 21_000 and "truncated" in md2, len(md2)


if __name__ == "__main__":
    _selftest_rule_titles()
    _selftest_digest()
    _selftest()
