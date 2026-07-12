"""holographic_codetriage.py -- honest triage of code in an UNRECOGNIZED language (backlog C5).

When source arrives in a language leCore has no parser for (holographic_codeparse covers only the emit-grammar
dialects), the honest move is NOT to guess a grammar from one sample -- grammar induction from a single example is
not knowledge, it is a hallucination with statistics stapled on. The honest move is to report STRUCTURAL
OBSERVATIONS that are true of the bytes regardless of language, and to LABEL them as observations, not
understanding. That boundary is the whole design: everything this module says is checkable against the source
character by character, and it never claims to know what the code DOES.

What it observes (all language-agnostic, all deterministic):
  - line/character counts and the comment lines it can find under the common comment conventions;
  - identifiers, extracted by the universal rule (letter/underscore then alphanumerics), then SPLIT on camelCase
    and snake_case into their word pieces and ranked by frequency -- the words a programmer chose are the single
    most information-dense signal in unfamiliar code (Deissenboeck & Pizka 2006, "Concise and Consistent Naming");
  - literals: numbers and quoted strings (single, double, backtick), counted, a few shown;
  - a nesting profile from bracket/brace/paren balance -- max depth and whether the brackets balance at all;
  - a WEAK language guess, offered only as a hint with its evidence, never as a conclusion.

KEPT NEGATIVE (the load-bearing one): this is TRIAGE, not comprehension. The output is a set of observations a
human can use to decide what the code probably is and whether to read it closely; it is not an explanation and
must never be dressed up as one. holographic_codeverbal explains code we can PARSE; this describes code we cannot.
The language guess in particular is explicitly weak -- it pattern-matches surface tokens (`def`/`fn`/`function`,
`;` density, `{` usage) and reports the evidence so a human can overrule it instantly. A confident guess from
surface tokens is exactly the overreach the module exists to avoid.

KEPT NEGATIVE (scope): comment detection uses the common conventions (# , // , /* */ , -- , ; for lisp/asm). A
language whose comment syntax is none of these will have its comments counted as code -- stated, not hidden. The
identifier and literal extraction is likewise convention-based and will mis-slice a language that, say, allows
hyphens in identifiers (Lisp). Each observation therefore travels with the convention it assumed.
"""

import re
from collections import Counter

#: comment openers we recognize. A language outside this set has its comments counted as code (stated negative).
_LINE_COMMENTS = ("#", "//", "--", ";")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_STRING = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|`[^`\n]*`')

#: keywords used ONLY as weak surface evidence for the language hint -- never as a parse.
_LANG_HINTS = {
    "python": (re.compile(r"\bdef\s+\w+\s*\("), re.compile(r":\s*$", re.MULTILINE)),
    "c-family": (re.compile(r"\b(?:int|float|double|void)\s+\w+\s*\("), re.compile(r";\s*$", re.MULTILINE)),
    "javascript": (re.compile(r"\bfunction\s+\w+\s*\("), re.compile(r"\b(?:const|let|var)\b")),
    "rust/zig": (re.compile(r"\bfn\s+\w+\s*\("), re.compile(r"\b(?:let|const|pub)\b")),
    "lisp-family": (re.compile(r"\(\s*(?:def|define|lambda)\b"), re.compile(r"\)\s*\)")),
}


def _split_identifier(ident):
    """Split an identifier into lowercase word pieces on camelCase and snake_case boundaries.

    getHTTPResponse -> [get, http, response]; max_step_count -> [max, step, count]. This is what turns raw
    identifiers into the words a programmer actually chose, which is the densest clue in unfamiliar code."""
    parts = ident.split("_")
    words = []
    for part in parts:
        if not part:
            continue
        # camelCase / PascalCase / acronym runs: split before a capital that starts a new word
        pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
        words.extend(pieces if pieces else [part])
    return [w.lower() for w in words if w]


def _comment_lines(lines):
    n = 0
    for ln in lines:
        s = ln.strip()
        if s and any(s.startswith(c) for c in _LINE_COMMENTS):
            n += 1
    return n


def _nesting_profile(src):
    """Max bracket-nesting depth and whether all three bracket kinds balance. Language-agnostic: it counts
    ()[]{} ignoring those inside detected strings (a best effort -- a string convention we do not know will leak,
    which is why the balance flag is reported, not asserted)."""
    stripped = _STRING.sub("", src)
    depth = maxdepth = 0
    counts = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0}
    for ch in stripped:
        if ch in "([{":
            depth += 1
            maxdepth = max(maxdepth, depth)
            counts[ch] += 1
        elif ch in ")]}":
            depth -= 1
            counts[ch] += 1
    balanced = (counts["("] == counts[")"] and counts["["] == counts["]"] and counts["{"] == counts["}"])
    return maxdepth, balanced


def _language_hint(src):
    """A WEAK guess with its evidence. Returns (best_guess_or_None, [(lang, n_signals), ...]). Never a parse."""
    scores = []
    for lang, pats in _LANG_HINTS.items():
        n = sum(1 for p in pats if p.search(src))
        if n:
            scores.append((lang, n))
    scores.sort(key=lambda x: (-x[1], x[0]))
    best = scores[0][0] if scores and scores[0][1] >= 1 else None
    return best, scores


def triage(src, top_words=12, top_strings=5):
    """Structural observations about code in an unrecognized language. Returns a dict of OBSERVATIONS (not an
    explanation): line/char/comment counts, ranked identifier word-pieces, literal inventory, nesting profile,
    and a weak language hint with its evidence. Every field is checkable against the source; none claims to know
    what the code does. This is triage; holographic_codeverbal is comprehension (for languages we can parse)."""
    src = str(src)
    lines = src.splitlines()
    body = _BLOCK_COMMENT.sub("", src)                    # drop /*...*/ so block comments don't pollute idents

    idents = _IDENT.findall(body)
    word_counter = Counter()
    for ident in idents:
        word_counter.update(_split_identifier(ident))
    numbers = _NUMBER.findall(body)
    strings = _STRING.findall(src)                        # strings from the ORIGINAL (block-comment strip is code)
    maxdepth, balanced = _nesting_profile(src)
    guess, evidence = _language_hint(src)

    return {
        "is_observation_not_explanation": True,          # the contract, made explicit in the payload
        "lines": len(lines),
        "chars": len(src),
        "comment_lines": _comment_lines(lines),
        "identifier_count": len(idents),
        "unique_identifiers": len(set(idents)),
        "top_word_pieces": word_counter.most_common(top_words),
        "number_literals": len(numbers),
        "string_literals": len(strings),
        "sample_strings": strings[:top_strings],
        "max_nesting_depth": maxdepth,
        "brackets_balanced": balanced,
        "language_hint": guess,
        "language_evidence": evidence,
        "caveats": ["comment detection assumed conventions %s and /*...*/; other conventions count as code"
                    % (_LINE_COMMENTS,),
                    "identifiers assumed [A-Za-z_][A-Za-z0-9_]* -- a language allowing e.g. hyphens will mis-slice",
                    "the language hint is a WEAK surface-token guess with its evidence attached, not a parse"],
    }


def triage_report(src):
    """A human-readable rendering of triage(): the same observations as prose lines, each honest about being an
    observation. For an agent that wants text rather than a dict."""
    t = triage(src)
    words = ", ".join("%s(%d)" % (w, n) for w, n in t["top_word_pieces"])
    hint = "%s (evidence: %s)" % (t["language_hint"], t["language_evidence"]) if t["language_hint"] \
        else "no confident surface-token match"
    return ("UNRECOGNIZED-LANGUAGE TRIAGE (observations, not an explanation):\n"
            "  size: %d lines, %d chars, %d comment lines (by convention)\n"
            "  identifiers: %d total, %d unique; top word pieces: %s\n"
            "  literals: %d numbers, %d strings %s\n"
            "  structure: max nesting depth %d, brackets %s\n"
            "  weak language hint: %s\n"
            "  caveat: this describes the bytes; it does NOT claim to know what the code does."
            % (t["lines"], t["chars"], t["comment_lines"], t["identifier_count"], t["unique_identifiers"],
               words, t["number_literals"], t["string_literals"],
               ("e.g. %r" % t["sample_strings"][0] if t["sample_strings"] else ""),
               t["max_nesting_depth"], "balanced" if t["brackets_balanced"] else "UNBALANCED (partial snippet?)",
               hint))


def _selftest():
    """Exact traps: identifier splitting on both conventions; ranked word pieces; literal counts; nesting depth;
    the weak hint fires with evidence AND is labelled weak; and the contract flag is present. A deliberately
    non-emit-grammar sample (a made-up C-ish language) is used so nothing here overlaps codeparse's job."""
    assert _split_identifier("getHTTPResponse") == ["get", "http", "response"], _split_identifier("getHTTPResponse")
    assert _split_identifier("max_step_count") == ["max", "step", "count"]
    assert _split_identifier("XMLParser2") == ["xml", "parser", "2"]

    sample = (
        "// a made-up language sample\n"
        "func computeStepBudget(maxDepth: Int, ray_origin: Vec3) -> Float {\n"
        "    let step_size = 0.001;\n"
        "    var total = 0.0;\n"
        "    loop while (total < maxDepth) {\n"
        "        total = total + step_size * 2.0;\n"
        '        log("stepping");\n'
        "    }\n"
        "    return total;\n"
        "}\n"
    )
    t = triage(sample)
    assert t["is_observation_not_explanation"] is True
    assert t["comment_lines"] == 1
    assert ("step", 2) in t["top_word_pieces"] or any(w == "step" for w, _ in t["top_word_pieces"])
    assert t["number_literals"] >= 3 and t["string_literals"] == 1 and t["sample_strings"] == ['"stepping"']
    assert t["max_nesting_depth"] >= 2 and t["brackets_balanced"] is True
    # weak hint: this sample has fn-ish + let/var, should lean rust/zig-ish, WITH evidence, never asserted certain
    assert t["language_hint"] is not None and t["language_evidence"]
    assert any("WEAK" in c or "weak" in c for c in t["caveats"])

    # unbalanced snippet is reported, not crashed
    assert triage("func f() { return (a + ")["brackets_balanced"] is False

    rep = triage_report(sample)
    assert "does NOT claim to know what the code does" in rep and "observations, not an explanation" in rep

    print("OK: holographic_codetriage self-test passed (identifier splitting exact on camelCase/snake_case/acronym "
          "runs; word pieces ranked; %d numbers + 1 string counted; nesting depth %d, balance detected; weak "
          "language hint fires WITH evidence and is labelled weak; unbalanced snippet reported not crashed; the "
          "observation-not-explanation contract is in the payload and the report)"
          % (t["number_literals"], t["max_nesting_depth"]))


if __name__ == "__main__":
    _selftest()
