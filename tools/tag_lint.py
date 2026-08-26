"""tools/tag_lint.py -- catch io-kind tags that LIE, before a router believes them.

WHY (a downstream audit, with receipts). A ComfyUI node pack reported:

    m.suggest_pipeline("image", "mesh")
    -> [Denoise multi-way data (low-rank tensor prior), Aharonov-Bohm ring (magnetic flux phase), ...]

"That's a nonsense route... A wrong tag doesn't just fail to help -- it actively misleads Auto Route." They asked
for a lint. Reproducing it found TWO bugs compounding, and neither was a typo:

  1. FAKE EDGES. The edge builders read `consumes` x `produces` as a CROSS PRODUCT. That is right for a
     CONJUNCTIVE capability (transform_selection needs mesh AND selection AND transform, so "I hold a selection,
     can I reach a mesh?" is a real question) and WRONG for a POLYMORPHIC one: denoise_tensor takes an image OR a
     field and returns THE SAME KIND, so `image->field` is a conversion it cannot perform. The router escaped
     into field-space through that invented hop. Fixed by `polymorphic=True`; this lint keeps it fixed.
  2. THE REAL ROUTE WAS UNTAGGED. `image_to_mesh` / `depth_to_mesh` / `photo_to_3d` each say "image -> MESH" in
     their own first docstring line and declared nothing, so no typed image->mesh edge existed at all. The router
     had no honest answer available and took the dishonest one.

WHAT IT CHECKS (cheap, deterministic, no smoke-calling -- see the kept negative below)
  * LIAR      -- a tagged capability whose `method` is not callable on a live mind: an edge nothing can execute.
  * CROSSFAKE -- consumes and produces OVERLAP but `polymorphic` is unset, so the cross product invents
                 off-diagonal edges. Either the capability really converts between those kinds (say so by
                 splitting the tag) or it is polymorphic (say THAT). Silence is what produced the bug report.
  * UNTAGGED_CONVERTER -- a faculty whose docstring announces "A -> B" in io-kind words while declaring nothing.
                 This is the miss that let the fake edge win, and it is S7's drive queue, generated rather than
                 guessed at.

KEPT NEGATIVE (loud): the audit asked for "a smoke-called result's actual kind". NOT DONE, deliberately. Smoke
calling needs a valid input per kind for 100+ faculties; the fixtures would be a second engine to maintain, and a
wrong fixture reports a lie about a lie. The three static checks above caught the ENTIRE reported failure, so the
expensive half is unbought until a bug survives them. If one ever does, that is the evidence to build it on.

Exit 0 = clean. Non-zero = a tag a router would act on and regret.
"""
import os
import re
import sys

# Same as tools/skill_lint.py: this tool lives in tools/, the engine is one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: Docstring shapes that ANNOUNCE a conversion. Deliberately narrow -- an over-eager pattern would flood the S7
#: queue with false positives, and a queue nobody trusts is a queue nobody works.
_ARROW = re.compile(r"\b([a-z_]+)\s*(?:->|-->|to)\s*([a-z_]+)\b", re.I)


def _kind_words(kind):
    """The words a docstring plausibly uses for an io kind. 'sdf_scene' is also written 'scene'."""
    return {kind, kind.replace("_", " "), kind.split("_")[0]}


def audit(mind=None):
    """Every tag a router would act on and regret: {'liars', 'crossfake', 'untagged_converters'}."""
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=64, seed=0)
    from holographic.caching_and_storage.holographic_iokinds import IO_KINDS

    cat = mind._capability_catalog()
    caps = cat.all()

    liars, crossfake = [], []
    for c in caps:
        tagged = bool(getattr(c, "consumes", ()) or getattr(c, "produces", ()))
        if tagged and getattr(c, "method", None) and not callable(getattr(mind, c.method, None)):
            liars.append((c.name, c.method))
        # The rule is TIGHT on purpose: `consumes == produces` as sets, with more than one kind. That is the
        # exact shape of "takes any of these, returns THE SAME one" (denoise_tensor: image|field -> image|field),
        # where every off-diagonal pair is a conversion the code cannot perform.
        # KEPT NEGATIVE, measured: the first draft flagged any OVERLAP and produced 5 false positives --
        # select_edge_loop(mesh, selection)->selection, skin_mesh(mesh, skeleton)->mesh,
        # transform_selection(mesh, selection, transform)->mesh. Those are CONJUNCTIVE (they need every consumed
        # kind at once), so "mesh->selection" is a REAL routing question and the cross product is right for them.
        # Overlap alone means nothing; a lint that cries wolf on legitimate tags would get muted, and then the
        # one real liar rides in behind it.
        cons, prod = set(getattr(c, "consumes", ())), set(getattr(c, "produces", ()))
        if len(cons) > 1 and cons == prod and not getattr(c, "polymorphic", False):
            invented = sorted("%s->%s" % (a, b) for a in cons for b in prod if a != b)
            crossfake.append((c.name, invented))

    # UNTAGGED_CONVERTER: a faculty announcing "A -> B" in kind words while declaring nothing.
    untagged = []
    by_name = {c.name: c for c in caps}
    for name in dir(mind):
        if name.startswith("_"):
            continue
        cap = by_name.get(name)
        if cap is None or cap.consumes or cap.produces:
            continue
        attr = getattr(type(mind), name, None)
        if not callable(attr):
            continue
        import inspect
        first = (inspect.getdoc(attr) or "").strip().split("\n")[0].lower()
        for a, b in _ARROW.findall(first):
            ka = next((k for k in IO_KINDS if a in _kind_words(k)), None)
            kb = next((k for k in IO_KINDS if b in _kind_words(k)), None)
            if ka and kb and ka != kb:
                untagged.append((name, "%s->%s" % (ka, kb), first[:64]))
                break
    return {"liars": liars, "crossfake": crossfake, "untagged_converters": untagged}


def main(argv):
    quiet = "--quiet" in argv
    r = audit()
    fail = len(r["liars"]) + len(r["crossfake"])

    if r["liars"]:
        print("LIARS (tagged, but mind.<method> is not callable -- an edge nothing can execute): %d" % len(r["liars"]))
        for n, meth in r["liars"]:
            print("    %-52s method=%r" % (n[:52], meth))
    if r["crossfake"]:
        print("CROSSFAKE (consumes/produces overlap, `polymorphic` unset -- the cross product INVENTS these): %d"
              % len(r["crossfake"]))
        for n, inv in r["crossfake"]:
            print("    %-52s invents %s" % (n[:52], inv))
        print("  FIX: pass polymorphic=True if it returns the kind it was given; otherwise split the tag so the")
        print("       real conversions are declared and the impossible ones are not.")
    if r["untagged_converters"] and not quiet:
        print("UNTAGGED CONVERTERS (docstring announces a conversion, tag declares nothing) -- S7's queue: %d"
              % len(r["untagged_converters"]))
        for n, arrow, doc in r["untagged_converters"][:20]:
            print("    %-24s %-18s %s" % (n, arrow, doc))
        print("  These are NOT failures -- they are the drive list. Every one tagged is a typed edge, and a")
        print("  downstream typed node, for free.")

    if not fail:
        print("OK: no lying io tags (0 liars, 0 crossfake). %d untagged converter(s) queued for S7."
              % len(r["untagged_converters"]))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
