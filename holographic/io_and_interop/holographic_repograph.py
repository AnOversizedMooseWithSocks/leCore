"""Structural repo graph: multi-language mapping, diagrams, grounded spec conformance.

WHY THIS MODULE EXISTS, AND WHY IT IS NOT holographic_codemap. codemap answers
the SIMILARITY question ("what else looks like this function") over python defs.
codebase_map (unified_p20_zoo, cp28) archives per-module PROSE for bm25 recall.
Neither answers the STRUCTURAL questions a code agent asks about a mixed-language
tree: which symbols exist across .py/.js/.c, which files lean on which, which
files matter most, and how to draw it. A find_capability audit for those
phrasings returned only fallbacks -- that absence is this module's license.

The construction follows the strongest public system (aider's repo map: per-file
defs + references -> file graph -> PageRank-style rank -> budgeted skeleton),
under leCore's constraints: stdlib + NumPy only, no tree-sitter, deterministic.

THE THREE JOBS:
  * RepoGraph    -- extract symbols per file (py via ast; js/c via conservative
                    lexing), build the def/ref file graph, rank files with a
                    fixed-iteration power-method PageRank, emit a budgeted
                    skeleton of the tree.
  * diagram()    -- the same graph as Mermaid or Graphviz DOT text. A diagram
                    that is text diffs, versions, and needs no renderer.
  * SpecChecker  -- take an SOP/spec, split it into atomic claims, verify each
                    against the tree with MECHANICAL evidence (file:line whose
                    text is re-read from disk and re-matched before it may be
                    cited). No evidence -> UNVERIFIABLE, an honest abstain,
                    never a guess: the checker can only cite what a re-read
                    can confirm, so it cannot hallucinate.

KEPT NEGATIVES, ON RECORD:
  * Regex "parsing" of full C/JS grammars was tried and rejected -- brace
    matching across preprocessor branches and JS ASI both defeat it. The
    extractors are deliberately CONSERVATIVE lexers: they claim a definition
    only when the line shape is unambiguous; everything else is a reference
    token. Missing a def costs a little rank precision; inventing one poisons
    the graph.
  * An LLM may PROPOSE claim decompositions in SpecChecker (llm= accepted) but
    never CONFIRM: every claim, whoever wrote it, passes the same mechanical
    evidence gate (the CiteCheck lesson -- the model is a comparator over
    retrieved evidence, never the authority).
  * Silent truncation is this codebase's named failure class (NOTES, sweeps
    59-60: grep widening + capped hits read as a clean result). Every cap in
    this module emits a loud marker in its output.
"""

import ast
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# language tables
# ---------------------------------------------------------------------------

# Keywords are excluded from reference tokens so graph edges count only
# identifiers that could plausibly be defined elsewhere in the tree.
_JS_KEYWORDS = frozenset("""
abstract arguments await boolean break byte case catch char class const
continue debugger default delete do double else enum eval export extends
false final finally float for function get goto if implements import in
instanceof int interface let long native new null of package private
protected public return set short static super switch synchronized this
throw throws transient true try typeof undefined var void volatile while
with yield async console module require exports
""".split())

_C_KEYWORDS = frozenset("""
auto break case char const continue default do double else enum extern float
for goto if inline int long register restrict return short signed sizeof
static struct switch typedef union unsigned void volatile while bool true
false NULL include define ifdef ifndef endif pragma undef elif error
""".split())

_PY_KEYWORDS = frozenset("""
False None True and as assert async await break class continue def del elif
else except finally for from global if import in is lambda nonlocal not or
pass print raise return self try while with yield
""".split())

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

# One conservative definition pattern per non-python language. Each fires only
# on a line shape that cannot reasonably be anything but a definition -- see
# the kept negative about regex parsing in the module docstring.
_JS_DEF = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?"
    r"(?:(?:async\s+)?function\s*\*?\s*(?P<fn>[A-Za-z_$][\w$]*)"        # function foo(
    r"|class\s+(?P<cls>[A-Za-z_$][\w$]*)"                                # class Foo
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*=\s*"            # const foo = (..) =>
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>))")
_C_DEF = re.compile(
    # return-type identifier(args) at line start, not ending in ';' (that is a
    # prototype -- indexed too, tagged 'proto', so declaration and definition
    # stay distinguishable).
    r"^[A-Za-z_][\w\s\*]*?\b(?P<fn>[A-Za-z_]\w*)\s*\([^;{]*\)\s*(?P<body>\{)?\s*$")
_C_MACRO = re.compile(r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)")
_C_TYPEDEF = re.compile(r"^\s*typedef\b.*?\b(?P<name>[A-Za-z_]\w*)\s*;\s*$")

_LANG_OF_SUFFIX = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".jsx": "javascript",
    ".c": "c", ".h": "c",
}

TRUNCATION_MARKER = "... TRUNCATED ..."

_SKIP_DIRS = ("__pycache__", ".git", "node_modules", ".venv", "venv",
              "archive", "dist", "build")


def language_of(path):
    """Language name for a path, or None when this module has no extractor.
    The suffix table is the contract: adding a language means one suffix here
    and one extractor below -- nothing else changes."""
    return _LANG_OF_SUFFIX.get(os.path.splitext(path)[1].lower())


# ---------------------------------------------------------------------------
# per-language symbol extraction
# ---------------------------------------------------------------------------

_IDENTCHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


def _scrub_for_anchors(text, rel):
    """Comment/string-scrubbed text for ANCHOR MATCHING, line numbers
    preserved. Python gets a per-line scrub (paired quotes blanked, then
    `#` onward dropped); c-like files reuse _strip_c_like_noise; anything
    else passes through raw. Deliberately conservative -- see _find's
    docstring: a violation hidden by exotic quoting stays hidden (false
    negative), which is the acceptable direction of error here."""
    low = rel.lower()
    if low.endswith((".js", ".c", ".h")):
        return _strip_c_like_noise(text)
    if not low.endswith(".py"):
        return text
    # EXACT python scrub via tokenize: the per-line fallback below cannot see
    # triple-quote state, so docstring INTERIORS leaked (this module's own
    # docstring mentioning `eval(` was cited as a violation of "never eval").
    # tokenize knows; COMMENT and STRING tokens are blanked in place, line
    # structure preserved. Broken files fall through to the per-line scrub.
    try:
        import io as _io
        import tokenize as _tk
        lines = text.split("\n")
        for tok in _tk.generate_tokens(_io.StringIO(text).readline):
            if tok.type not in (_tk.COMMENT, _tk.STRING):
                continue
            (r0, c0), (r1, c1) = tok.start, tok.end
            for rr in range(r0 - 1, r1):
                a = c0 if rr == r0 - 1 else 0
                b = c1 if rr == r1 - 1 else len(lines[rr])
                lines[rr] = lines[rr][:a] + " " * (b - a) + lines[rr][b:]
        return "\n".join(lines)
    except Exception:
        pass
    out = []
    for line in text.split("\n"):
        buf, i, n = [], 0, len(line)
        while i < n:
            ch = line[i]
            if ch in "\"'":
                q = line.find(ch, i + 1)
                while q != -1 and line[q - 1] == "\\":
                    q = line.find(ch, q + 1)
                if q == -1:               # unterminated on this line: docstring
                    break                 # interior or continuation -- drop rest
                buf.append(" " * (q - i + 1))
                i = q + 1
                continue
            if ch == "#":
                break
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def _anchor_on_boundary(line, anchor):
    """True when `anchor` occurs in `line` on a token boundary: if the anchor
    STARTS with an identifier character, the character before the hit may not
    be an identifier character or '.' (so `eval(` refuses `model.eval(` and
    `series_eval(`); if it ENDS with one, the character after may not be an
    identifier character. Anchors with non-identifier edges keep plain
    substring semantics."""
    start = 0
    while True:
        j = line.find(anchor, start)
        if j == -1:
            return False
        ok = True
        if anchor[0] in _IDENTCHARS and j > 0 and \
                (line[j - 1] in _IDENTCHARS or line[j - 1] == "."):
            ok = False
        end = j + len(anchor)
        if ok and anchor[-1] in _IDENTCHARS and end < len(line) and \
                line[end] in _IDENTCHARS:
            ok = False
        if ok:
            return True
        start = j + 1


def _strip_c_like_noise(src):
    """Remove comments and string literals from C/JS source before lexing.
    WHY: a reference counted inside a comment or string is not a dependency;
    leaving them in adds edges between files that merely mention each other in
    prose. Newlines are preserved so line numbers survive the strip."""
    out, i, n = [], 0, len(src)
    while i < n:
        two = src[i:i + 2]
        c = src[i]
        if two == "//":                                   # line comment
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":                                 # block comment
            j = src.find("*/", i + 2)
            seg = src[i:(n if j < 0 else j + 2)]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        elif c in "\"'`":                                 # string / template
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            seg = src[i:j + 1]
            out.append("\n" * seg.count("\n"))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract_python(src):
    """Symbols of one python module: (defs, refs, imports).
    defs: [(name, kind, line, signature)] for top-level and class-level
    functions/classes -- the public surface an agent needs to see.
    refs: every Name/Attribute head anywhere (minus keywords) -- raw material
    of the dependency graph. imports: top-level module names."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], sorted(set(_IDENT.findall(src)) - _PY_KEYWORDS), []
    defs, imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = "(%s)" % ", ".join(a.arg for a in node.args.args)
            defs.append((node.name, "function", node.lineno, sig))
        elif isinstance(node, ast.ClassDef):
            defs.append((node.name, "class", node.lineno, ""))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = "(%s)" % ", ".join(a.arg for a in sub.args.args)
                    defs.append((node.name + "." + sub.name, "method",
                                 sub.lineno, sig))
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)
        elif isinstance(node, ast.Import):
            imports += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return defs, sorted(refs - _PY_KEYWORDS), sorted(set(imports))


def extract_javascript(src):
    """Symbols of one JS module via the conservative lexer. Imports cover both
    `import ... from 'x'` and `require('x')`; only the bare module head is
    kept ('./util' -> 'util') so it can match a sibling file's stem."""
    clean = _strip_c_like_noise(src)
    defs = []
    for lineno, line in enumerate(clean.split("\n"), 1):
        m = _JS_DEF.match(line)
        if not m:
            continue
        name = m.group("fn") or m.group("cls") or m.group("var")
        kind = "class" if m.group("cls") else "function"
        defs.append((name, kind, lineno, ""))
    imports = set()
    for m in re.finditer(r"""(?:from|require\s*\()\s*['"]([^'"]+)['"]""", src):
        head = m.group(1).rstrip("/").split("/")[-1]
        imports.add(os.path.splitext(head)[0])
    refs = sorted(set(_IDENT.findall(clean)) - _JS_KEYWORDS)
    return defs, refs, sorted(imports)


def extract_c(src):
    """Symbols of one C translation unit. A function definition requires an
    opening brace (a trailing ';' is a prototype, kept but tagged 'proto');
    macros and typedefs are first-class defs because in real C trees they ARE
    the public surface."""
    clean = _strip_c_like_noise(src)
    defs = []
    lines = clean.split("\n")
    for lineno, line in enumerate(lines, 1):
        m = _C_MACRO.match(line)
        if m:
            defs.append((m.group("name"), "macro", lineno, ""))
            continue
        m = _C_TYPEDEF.match(line)
        if m:
            defs.append((m.group("name"), "typedef", lineno, ""))
            continue
        m = _C_DEF.match(line)
        if m and m.group("fn") not in _C_KEYWORDS:
            # Definition if a brace opens on this line or the next non-blank
            # one; wrong only for styles that put the brace 2+ lines down --
            # accepted, conservative (miss < invent).
            has_body = bool(m.group("body"))
            if not has_body:
                for nxt in lines[lineno:lineno + 2]:
                    if nxt.strip():
                        has_body = nxt.lstrip().startswith("{")
                        break
            defs.append((m.group("fn"),
                         "function" if has_body else "proto", lineno, ""))
    imports = sorted({os.path.splitext(os.path.basename(m.group(1)))[0]
                      for m in re.finditer(r'#\s*include\s*[<"]([^>"]+)[>"]',
                                           src)})
    refs = sorted(set(_IDENT.findall(clean)) - _C_KEYWORDS)
    return defs, refs, imports


_EXTRACTORS = {"python": extract_python,
               "javascript": extract_javascript,
               "c": extract_c}


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

class RepoGraph:
    """One scanned tree: files, symbols, the def/ref graph, and its ranking.

    Determinism contract: same tree bytes -> same graph, same ranks, same
    skeleton, byte for byte. Guaranteed by sorted walks, fixed-iteration
    PageRank from a uniform start (no randomness anywhere), and rank ties
    broken by relpath.
    """

    def __init__(self, root, max_files=20000, max_bytes=2_000_000):
        self.root = str(root)
        self.files = {}        # rel -> {lang, defs, refs, imports, lines}
        self.def_owner = {}    # bare symbol -> sorted list of rel paths
        self.truncated = False
        self._scan(max_files, max_bytes)
        self._build_graph()

    # -- scanning ----------------------------------------------------------
    def _scan(self, max_files, max_bytes):
        count = 0
        for r, dd, ff in os.walk(self.root):
            dd[:] = sorted(d for d in dd if d not in _SKIP_DIRS
                           and not d.startswith("."))
            for f in sorted(ff):
                lang = language_of(f)
                if lang is None:
                    continue
                if count >= max_files:
                    self.truncated = True      # loud, never silent
                    return
                path = os.path.join(r, f)
                rel = os.path.relpath(path, self.root)
                try:
                    if os.path.getsize(path) > max_bytes:
                        self.truncated = True
                        continue
                    src = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                defs, refs, imports = _EXTRACTORS[lang](src)
                self.files[rel] = {"lang": lang, "defs": defs, "refs": refs,
                                   "imports": imports,
                                   "lines": src.count("\n") + 1}
                count += 1

    # -- graph -------------------------------------------------------------
    def _build_graph(self):
        """Edges: file A -> file B, weighted by how many of A's reference
        tokens are names B defines -- aider's def/ref construction with the
        tokenizer standing in for tree-sitter captures."""
        owners = {}
        for rel, info in self.files.items():
            for name, _kind, _ln, _sig in info["defs"]:
                head = name.split(".")[-1]     # method refs arrive unqualified
                owners.setdefault(head, set()).add(rel)
        self.def_owner = {k: sorted(v) for k, v in owners.items()}
        self.paths = sorted(self.files)
        idx = {p: i for i, p in enumerate(self.paths)}
        n = len(self.paths)
        W = np.zeros((n, n))
        for rel, info in self.files.items():
            i = idx[rel]
            for tok in info["refs"]:
                for owner in owners.get(tok, ()):
                    if owner != rel:
                        W[i, idx[owner]] += 1.0
        self.adjacency = W
        self.rank = self._pagerank(W)

    @staticmethod
    def _pagerank(W, damping=0.85, iters=60, teleport=None):
        """Power iteration, FIXED iteration count, optional PERSONALIZATION.
        WHY fixed instead of tolerance-stopped: a tolerance stop makes the
        result depend on float summation order across BLAS builds; 60 rounds
        is past convergence for any tree we scan and identical everywhere.
        `teleport` (sweep 65, the aider-style focus bias from the SOTA
        survey): a per-node restart distribution -- rank mass restarts at the
        FOCUS files instead of uniformly, so files structurally NEAR the work
        at hand outrank globally-popular-but-irrelevant ones. None keeps the
        classic uniform restart bit-for-bit."""
        n = W.shape[0]
        if n == 0:
            return np.zeros(0)
        out = W.sum(axis=1, keepdims=True)
        T = np.divide(W, out, out=np.full_like(W, 1.0 / n),
                      where=out > 0)                 # dangling -> uniform
        if teleport is None:
            v = np.full(n, 1.0 / n)
        else:
            v = np.asarray(teleport, float)
            s = v.sum()
            v = v / s if s > 0 else np.full(n, 1.0 / n)
        r = v.copy()
        for _ in range(iters):
            r = (1 - damping) * v + damping * (T.T @ r)
        return r

    def refocus(self, focus, boost=50.0):
        """RE-RANK around the work at hand (personalized PageRank): `focus`
        is a list of relpaths (or basename fragments) the caller is editing
        or asking about; those nodes get `boost` x teleport weight (aider's
        measured 50x chat-file multiplier), everything else weight 1. Updates
        self.rank in place and returns the matched focus relpaths -- an
        UNMATCHED focus list is returned empty and the rank is left classic,
        never silently rebiased toward nothing."""
        weights = np.ones(len(self.paths))
        matched = []
        for i, rel in enumerate(self.paths):
            for f in (focus or ()):
                f = str(f)
                if rel == f or rel.endswith("/" + f) or os.path.basename(rel) == f:
                    weights[i] = float(boost)
                    matched.append(rel)
                    break
        if matched:
            self.rank = self._pagerank(self.adjacency, teleport=weights)
        return sorted(matched)

    def ranked_files(self):
        """[(rel, rank)] best first, ties broken by path -- the deterministic
        ordering everything downstream (skeleton, diagram) inherits."""
        order = sorted(range(len(self.paths)),
                       key=lambda i: (-self.rank[i], self.paths[i]))
        return [(self.paths[i], float(self.rank[i])) for i in order]

    # -- outputs -----------------------------------------------------------
    def skeleton(self, budget_lines=200):
        """The budgeted repo map: most-referenced files first, each with its
        definitions -- the aider insight that a ranked skeleton beats both
        'dump everything' and 'hand-pick files'. Truncation is announced."""
        out = ["REPO MAP  root=%s  files=%d  langs=%s" %
               (self.root, len(self.files),
                ",".join(sorted({v["lang"] for v in self.files.values()})))]
        for rel, rank in self.ranked_files():
            info = self.files[rel]
            head = "%s  [%s, %d lines, rank %.4f]" % (
                rel, info["lang"], info["lines"], rank)
            block = [head] + ["    %s %s%s  :%d" % (kind, name, sig, ln)
                              for name, kind, ln, sig in info["defs"][:40]]
            if len(info["defs"]) > 40:
                block.append("    " + TRUNCATION_MARKER +
                             " %d more defs" % (len(info["defs"]) - 40))
            if len(out) + len(block) > budget_lines:
                out.append(TRUNCATION_MARKER +
                           " budget %d lines reached" % budget_lines)
                break
            out += block
        if self.truncated:
            out.append(TRUNCATION_MARKER + " scan hit max_files/max_bytes cap")
        return "\n".join(out)

    def summary(self):
        """Machine-shaped digest for callers that want numbers, not prose."""
        langs = {}
        for v in self.files.values():
            langs[v["lang"]] = langs.get(v["lang"], 0) + 1
        return {"root": self.root, "files": len(self.files),
                "languages": langs,
                "defs": sum(len(v["defs"]) for v in self.files.values()),
                "edges": int((self.adjacency > 0).sum()),
                "truncated": self.truncated,
                "top": [p for p, _ in self.ranked_files()[:10]]}


# ---------------------------------------------------------------------------
# diagrams
# ---------------------------------------------------------------------------

def diagram(graph, fmt="mermaid", max_nodes=24, max_edges=48):
    """The file graph as diagram TEXT (mermaid flowchart or graphviz dot).
    Top-ranked files become nodes; the heaviest edges among them are drawn,
    labeled with reference counts. Caps are announced in a note node -- never
    silently (the named failure class)."""
    ranked = graph.ranked_files()[:max_nodes]
    keep = {rel for rel, _ in ranked}
    idx = {p: i for i, p in enumerate(graph.paths)}
    edges = []
    for a in keep:
        for b in keep:
            if a != b:
                w = graph.adjacency[idx[a], idx[b]]
                if w > 0:
                    edges.append((a, b, int(w)))
    edges.sort(key=lambda e: (-e[2], e[0], e[1]))
    dropped = max(0, len(edges) - max_edges)
    edges = edges[:max_edges]
    nid = {rel: "n%d" % i for i, (rel, _) in enumerate(ranked)}
    if fmt == "mermaid":
        lines = ["flowchart LR"]
        for rel, _ in ranked:
            lines.append('    %s["%s"]' % (nid[rel], rel.replace('"', "'")))
        for a, b, w in edges:
            lines.append("    %s -->|%d| %s" % (nid[a], w, nid[b]))
        if dropped or len(graph.files) > max_nodes:
            lines.append('    note["%s %d edges / %d files not drawn"]' %
                         (TRUNCATION_MARKER, dropped,
                          max(0, len(graph.files) - max_nodes)))
        return "\n".join(lines)
    if fmt == "dot":
        lines = ["digraph repo {", "  rankdir=LR;", "  node [shape=box];"]
        for rel, _ in ranked:
            lines.append('  %s [label="%s"];' % (nid[rel],
                                                 rel.replace('"', "'")))
        for a, b, w in edges:
            lines.append('  %s -> %s [label="%d"];' % (nid[a], nid[b], w))
        if dropped or len(graph.files) > max_nodes:
            lines.append('  trunc [shape=note, label="%s"];'
                         % TRUNCATION_MARKER)
        lines.append("}")
        return "\n".join(lines)
    raise ValueError("fmt must be 'mermaid' or 'dot', got %r" % (fmt,))


# ---------------------------------------------------------------------------
# grounded spec conformance
# ---------------------------------------------------------------------------

_ANCHOR_BACKTICK = re.compile(r"`([^`]+)`")
_ANCHOR_QUOTED = re.compile(r"\"([^\"]{2,60})\"|'([^']{2,60})'")
_ANCHOR_FILE = re.compile(r"\b[\w./-]+\.(?:py|js|mjs|cjs|jsx|c|h|md|json"
                          r"|txt|toml|yaml|yml|sh)\b")
_ANCHOR_IDENT = re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]+"      # snake_case
                           r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"  # CamelCase
                           r"|[a-z]+[A-Z][A-Za-z0-9]*)\b")        # camelCase

_NEGATION = re.compile(r"\b(never|not|no|must not|shall not|forbidden|"
                       r"disallowed|without)\b", re.I)


class SpecChecker:
    """Verify an SOP/spec against a source tree, claim by claim, with
    mechanically re-confirmed evidence -- and honest abstention where the
    text gives the machine nothing to grip.

    THE GRIP RULE. A sentence is checkable only if it contains an ANCHOR: a
    backticked term, a quoted string, a filename, or a code-shaped identifier
    (snake_case / CamelCase / camelCase). Prose without anchors ('the system
    shall be robust') is returned UNVERIFIABLE, not judged -- pretending to
    verify it is exactly the hallucination this tool exists to prevent.

    VERDICTS per claim (tri-state plus polarity):
      supported     -- every anchor found; [file:line] cited, each citation
                       re-read from disk and re-matched before it may appear.
      partial       -- some anchors found; found and missing sets both
                       reported, loudly.
      unverifiable  -- no anchors, or none found: an abstain, never a 'false'.
      violated      -- the claim NEGATES its anchors ('never use X') and an
                       anchor was found anyway; the evidence shows where.
    """

    def __init__(self, root, max_hits_per_anchor=5, max_search_files=8000):
        self.root = str(root)
        self.max_hits = max_hits_per_anchor
        self.max_search_files = max_search_files
        self._file_cache = None

    # -- claim decomposition ----------------------------------------------
    @staticmethod
    def claims_of(spec_text, llm=None):
        """Split spec text into atomic claims. Bullets and sentences are the
        atoms; an optional llm callable may propose a finer split, but its
        output re-enters the same anchor gate as everything else (propose vs
        confirm -- the module-level kept negative)."""
        if llm is not None:
            try:
                proposed = str(llm(
                    "Split into one atomic requirement per line, no "
                    "commentary, preserve exact identifiers:\n" + spec_text))
                lines = [l.strip("-* \t") for l in proposed.split("\n")]
                got = [l for l in lines if len(l) > 8]
                if got:
                    return got
            except Exception:
                pass  # a dead model degrades to the deterministic split
        atoms = []
        for raw in spec_text.split("\n"):
            line = raw.strip().lstrip("-*").strip()
            if not line or line.startswith("#"):
                continue
            # Split on sentence ends but never inside backticks.
            parts, buf, tick = [], [], False
            for ch in line:
                if ch == "`":
                    tick = not tick
                buf.append(ch)
                if ch in ".;" and not tick:
                    parts.append("".join(buf).strip(" .;"))
                    buf = []
            if buf:
                parts.append("".join(buf).strip(" .;"))
            atoms += [p for p in parts if len(p) > 8]
        return atoms

    @staticmethod
    def anchors_of(claim):
        """The checkable tokens of one claim, deduplicated, order-stable."""
        found = []
        for m in _ANCHOR_BACKTICK.finditer(claim):
            found.append(m.group(1).strip())
        for m in _ANCHOR_FILE.finditer(claim):
            found.append(m.group(0))
        for m in _ANCHOR_QUOTED.finditer(claim):
            found.append((m.group(1) or m.group(2)).strip())
        stripped = _ANCHOR_BACKTICK.sub(" ", claim)
        for m in _ANCHOR_IDENT.finditer(stripped):
            found.append(m.group(0))
        seen, out = set(), []
        for a in found:
            if a and a not in seen:
                seen.add(a)
                out.append(a)
        return out

    # -- evidence ----------------------------------------------------------
    def _source_files(self):
        if self._file_cache is None:
            acc = []
            for r, dd, ff in os.walk(self.root):
                dd[:] = sorted(d for d in dd if d not in _SKIP_DIRS
                               and not d.startswith("."))
                for f in sorted(ff):
                    if os.path.splitext(f)[1].lower() in (
                            ".py", ".js", ".mjs", ".cjs", ".jsx", ".c", ".h",
                            ".md", ".txt", ".toml", ".json", ".yaml", ".yml"):
                        acc.append(os.path.join(r, f))
                    if len(acc) >= self.max_search_files:
                        self._file_cache = acc
                        return acc
            self._file_cache = acc
        return self._file_cache

    def _find(self, anchor):
        """[(rel, line_no, line_text)] hits for one anchor, capped LOUDLY:
        a capped result carries ('...', -1, TRUNCATION_MARKER) as its final
        entry so absence-of-more is never mistaken for absence.

        PRECISION RULES (sweep 63, earned on real ground -- the leCore/leOS
        hazard audit): matching runs on comment/string-SCRUBBED text, and an
        anchor beginning or ending in an identifier character must sit on a
        token boundary. Both rules exist because the raw-substring version
        cited `# never hash()` prose and torch's `model.eval()` as
        violations of "never use hash(/eval(" -- an instrument that indicts
        the WARNING SIGNS is worse than none. False negatives (a violation
        hidden in exotic quoting) are acceptable; false accusations are not.
        The RAW line is still what gets cited -- the reader sees the file's
        truth, not the scrub."""
        hits = []
        for path in self._source_files():
            rel = os.path.relpath(path, self.root)
            if anchor == os.path.basename(rel) or anchor == rel:
                hits.append((rel, 0, "(filename match)"))
                continue
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            scrubbed = _scrub_for_anchors(text, rel)
            if anchor not in scrubbed:
                continue
            raw_lines = text.split("\n")
            for i, line in enumerate(scrubbed.split("\n"), 1):
                if _anchor_on_boundary(line, anchor):
                    hits.append((rel, i, raw_lines[i - 1].strip()[:160]))
                    if len(hits) >= self.max_hits:
                        hits.append(("...", -1, TRUNCATION_MARKER))
                        return hits
        return hits

    def _reconfirm(self, rel, line_no, anchor):
        """THE MECHANICAL GATE: re-read the cited line from disk and demand
        the anchor is literally on it. Evidence that fails re-reading is
        dropped -- a citation is a claim about a file, and the FILE is the
        authority, not the search that produced the citation."""
        if line_no == 0:      # filename-match evidence: the path is the proof
            return os.path.exists(os.path.join(self.root, rel))
        if line_no < 0:
            return False
        try:
            text = open(os.path.join(self.root, rel),
                        encoding="utf-8", errors="ignore").read()
            # Same scrub + boundary rule as _find: the gate must demand the
            # anchor in EXECUTABLE text, or a prose mention re-confirms itself.
            line = _scrub_for_anchors(text, rel).split("\n")[line_no - 1]
            return _anchor_on_boundary(line, anchor)
        except (OSError, IndexError):
            return False

    # -- the check ---------------------------------------------------------
    def check(self, spec_text, llm=None):
        """The full conformance report: per-claim verdicts with re-confirmed
        evidence, plus honest totals. `coverage` counts only claims the
        machine could actually grip -- unverifiable claims are surfaced,
        never folded into a percentage that would flatter the result."""
        claims = self.claims_of(spec_text, llm=llm)
        report, n_sup, n_checkable = [], 0, 0
        for claim in claims:
            anchors = self.anchors_of(claim)
            negated = bool(_NEGATION.search(claim))
            if not anchors:
                report.append({"claim": claim, "verdict": "unverifiable",
                               "why": "no checkable anchor (identifier, "
                                      "filename, quoted or backticked term)",
                               "evidence": {}, "missing": []})
                continue
            n_checkable += 1
            evidence, missing = {}, []
            for a in anchors:
                good = [(r, ln, tx) for (r, ln, tx) in self._find(a)
                        if self._reconfirm(r, ln, a)]
                if good:
                    evidence[a] = ["%s:%d  %s" % h for h in good]
                else:
                    missing.append(a)
            if negated:
                # 'never use X': found evidence is a VIOLATION, absence is
                # support. Polarity flips the reading, not the evidence.
                if evidence:
                    verdict = "violated"
                else:
                    verdict, n_sup = "supported", n_sup + 1
            elif not missing:
                verdict, n_sup = "supported", n_sup + 1
            elif evidence:
                verdict = "partial"
            else:
                verdict = "unverifiable"
                n_checkable -= 1        # nothing gripped after all
            report.append({"claim": claim, "verdict": verdict,
                           "evidence": evidence, "missing": missing})
        return {"root": self.root, "claims": len(claims),
                "checkable": n_checkable, "supported": n_sup,
                "coverage": (n_sup / n_checkable) if n_checkable else None,
                "report": report}


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="repograph_st_")
    try:
        # A tiny mixed tree with PLANTED dependency structure: util is used
        # by both main.py and app.js, so it must out-rank them.
        open(os.path.join(tmp, "util.py"), "w").write(
            '"""Shared helpers."""\n\ndef clamp(x, lo, hi):\n'
            '    return max(lo, min(hi, x))\n\nclass Store:\n'
            '    def get(self, k):\n        return k\n')
        open(os.path.join(tmp, "main.py"), "w").write(
            "from util import clamp, Store\n\ndef run():\n"
            "    s = Store()\n    return clamp(s.get(3), 0, 1)\n")
        open(os.path.join(tmp, "app.js"), "w").write(
            "// entry\nconst util = require('./util');\n"
            "function start() { return clamp(1, 0, 2); }\n"
            "const boot = () => start();\nclass App {}\n")
        open(os.path.join(tmp, "core.c"), "w").write(
            "#include <stdio.h>\n#define MAXN 16\n"
            "typedef struct { int x; } Node;\n"
            "int clamp_int(int x, int lo, int hi);\n"
            "int clamp_int(int x, int lo, int hi) {\n"
            "    return x < lo ? lo : (x > hi ? hi : x);\n}\n")

        g = RepoGraph(tmp)
        assert len(g.files) == 4, g.files.keys()
        # planted truth 1: every language extractor found its definitions
        py = {n: k for n, k, _l, _s in g.files["util.py"]["defs"]}
        assert py.get("clamp") == "function" and py.get("Store") == "class"
        assert ("Store.get", "method") in [(n, k) for n, k, _l, _s
                                           in g.files["util.py"]["defs"]]
        js = {n: k for n, k, _l, _s in g.files["app.js"]["defs"]}
        assert js.get("start") == "function", js       # function decl
        assert js.get("boot") == "function", js        # arrow const
        assert js.get("App") == "class", js
        cd = {n: k for n, k, _l, _s in g.files["core.c"]["defs"]}
        assert cd.get("MAXN") == "macro" and cd.get("Node") == "typedef"
        assert cd.get("clamp_int") == "function", cd   # definition wins proto
        # planted truth 2: the shared file out-ranks its consumers
        ranked = g.ranked_files()
        assert ranked[0][0] == "util.py", ranked
        # determinism: a second scan is byte- and bit-identical
        g2 = RepoGraph(tmp)
        assert g.skeleton(80) == g2.skeleton(80)
        assert np.array_equal(g.rank, g2.rank)
        # diagram: both formats draw the top file, and bad fmt refuses loudly
        assert "util.py" in diagram(g, "mermaid")
        assert "util.py" in diagram(g, "dot")
        try:
            diagram(g, "png")
            raise AssertionError("bad fmt must raise")
        except ValueError:
            pass
        # KEPT NEGATIVE pinned: comments/strings must NOT create references.
        # 'clamp' inside a js string would edge noise.js->util.py without the
        # strip; with it, only the genuine identifier remains.
        open(os.path.join(tmp, "noise.js"), "w").write(
            "// clamp Store start boot\nconst s = 'clamp Store';\n")
        g3 = RepoGraph(tmp)
        assert g3.files["noise.js"]["refs"] == ["s"], \
            g3.files["noise.js"]["refs"]

        # spec conformance: all four verdicts exercised on planted truth.
        spec = ("The module `util.py` provides `clamp` and `Store`.\n"
                "- Never use `subprocess_spawn` anywhere.\n"
                "- The system shall be excellent.\n"
                "- `clamp` lives beside `missing_thing_xyz`.\n")
        rep = SpecChecker(tmp).check(spec)
        verdicts = [r["verdict"] for r in rep["report"]]
        assert verdicts == ["supported", "supported", "unverifiable",
                            "partial"], verdicts
        assert rep["coverage"] is not None and 0.5 < rep["coverage"] <= 1.0
        # negation flip: plant the forbidden call, claim must flip to violated
        open(os.path.join(tmp, "bad.py"), "w").write(
            "def subprocess_spawn():\n    return 1\n")
        rep2 = SpecChecker(tmp).check("Never use `subprocess_spawn` anywhere.")
        assert rep2["report"][0]["verdict"] == "violated", rep2["report"][0]
        # every citation must re-confirm from disk (the mechanical gate)
        for r in rep2["report"]:
            for a, cites in r["evidence"].items():
                for c in cites:
                    rel, ln = c.split("  ")[0].rsplit(":", 1)
                    line = open(os.path.join(tmp, rel)).read().split("\n")[
                        int(ln) - 1]
                    assert a in line, (a, c)
        # PRECISION (sweep 63): prose, strings, attribute calls and
        # name-suffix lookalikes must NOT satisfy a negation claim -- only
        # executable use may. Earned on the leCore/leOS hazard audit, where
        # the raw matcher indicted "# never hash()" warnings and torch's
        # model.eval() as the violations they warn against.
        pdir = os.path.join(tmp, "prec")
        os.makedirs(pdir)
        open(os.path.join(pdir, "p.py"), "w").write(
            "# never call eval( here\n"
            "s = 'eval( in a string'\n"
            "model.eval()\n"
            "x = fourier_eval(3)\n"
            "y = eval(inp)\n")
        prep = SpecChecker(pdir).check("Never use `eval(` anywhere.")
        pcites = [c for v in prep["report"][0]["evidence"].values() for c in v]
        assert len(pcites) == 1 and pcites[0].startswith("p.py:5"), pcites
        print("holographic_repograph selftest OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
