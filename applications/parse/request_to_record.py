"""request_to_record -- turn a plain-English request into a structured record, and refuse when it can't.

WHAT THIS IS FOR: an agent handed "denoise a noisy mesh" needs {action, object, quality}, not a string.
mind.extract_roles() parses over a CONTROLLED VOCABULARY -- the object fillers are the engine's own
io-kinds -- so the record it returns is already in the vocabulary the router and the catalog speak.

THE PROPERTY WORTH DEMONSTRATING is the refusal, not the parse. Missing roles are OMITTED, never
fabricated, and an unparseable request returns {} rather than a plausible-looking guess. That is the
same abstention discipline the rest of this engine is built on, at the parsing layer: a router fed a
confident wrong record acts confidently and wrongly, which is worse than a router told nothing.

MEASURED, and asserted: 8/8 in-vocabulary requests yield a record with the expected action or object;
4/4 out-of-vocabulary requests ("recite a poem about the sea") return {} -- no fabricated roles, zero
false parses. Runtime is milliseconds; nothing is trained and nothing is loaded.

THE TRAP, worth the file on its own: this is NOT a general English role extractor, and reading it as one
is how you conclude it is broken. "the cat sat on the mat", "Alice gives Bob a book" and
"winch pulls crate" all return {} -- correctly, because none of their nouns are io-kinds. The vocabulary
is the engine's, so the sentences must be about the engine's work. Those exact strings are in the
out-of-vocabulary set below so the point is pinned rather than explained.
"""
NAME = "request_to_record"
DOMAIN = "parse"
PROVES = ("8/8 in-vocabulary engine requests parsed into {action, object, quality} records and 4/4 "
          "out-of-vocabulary ones refused with {} -- no fabricated roles")
ARTEFACT = None

#: (request, the role that must be present, its expected value). Deliberately spread across actions,
#: objects and qualities so a parser that only ever finds one role cannot pass.
IN_VOCAB = (
    ("denoise a noisy mesh", "object", "mesh"),
    ("denoise a noisy mesh", "quality", "noise"),
    ("render a scene to an image", "action", "render"),
    ("render a scene to an image", "object", "image"),
    ("smooth the mesh", "object", "mesh"),
    ("denoise a noisy image", "object", "image"),
    ("unwrap a mesh to uv", "object", "mesh"),
    ("encode text as a vector", "object", "hypervector"),
)

#: Requests the controlled vocabulary genuinely cannot serve. The first three are ordinary English
#: sentences that a general role extractor would happily parse -- which is exactly the misreading.
OUT_OF_VOCAB = (
    "the cat sat on the mat",
    "Alice gives Bob a book",
    "winch pulls crate",
    "recite a poem about the sea",
)


def run(mind, **_kw):
    """Parse every fixture request and count exact hits and honest refusals.

    Returns {rows, refusals, proved: {parsed, expected, refused, fabricated}}. `fabricated` is the one
    that must be zero: a role invented for a request the vocabulary cannot serve."""
    rows = []
    for text, role, want in IN_VOCAB:
        rec = mind.extract_roles(text)
        rows.append({"text": text, "role": role, "want": want, "got": rec.get(role), "record": rec,
                     "ok": rec.get(role) == want})
    refusals = []
    for text in OUT_OF_VOCAB:
        rec = mind.extract_roles(text)
        refusals.append({"text": text, "record": rec, "refused": rec == {}})
    return {"rows": rows, "refusals": refusals,
            "proved": {"parsed": sum(r["ok"] for r in rows), "expected": len(rows),
                       "refused": sum(r["refused"] for r in refusals),
                       "out_of_vocab": len(refusals),
                       "fabricated": sum(0 if r["refused"] else len(r["record"]) for r in refusals)}}


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    r = run(mind)
    p = r["proved"]
    # 1. Every in-vocabulary request parses to the EXPECTED role value -- not merely to something.
    assert p["parsed"] == p["expected"] == 8, [row for row in r["rows"] if not row["ok"]]
    # 2. THE PROPERTY: every out-of-vocabulary request is refused outright, and nothing is invented.
    assert p["refused"] == p["out_of_vocab"] == 4, r["refusals"]
    assert p["fabricated"] == 0, r["refusals"]
    # 3. The records must be genuinely structured -- at least one carries all three roles, or "parsed"
    #    could be satisfied by a parser that only ever fills a single field.
    assert any(len(row["record"]) == 3 for row in r["rows"]), [row["record"] for row in r["rows"]]
    # 4. Determinism, which the whole engine assumes: the same text gives the same record.
    assert mind.extract_roles("smooth the mesh") == mind.extract_roles("smooth the mesh")
    print("request_to_record OK: %d/%d requests parsed to the expected role, %d/%d out-of-vocabulary "
          "refused with {}, 0 roles fabricated"
          % (p["parsed"], p["expected"], p["refused"], p["out_of_vocab"]))


if __name__ == "__main__":
    _selftest()
