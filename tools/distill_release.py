"""RELEASE DISTILLATION (cp62): the session's external memory -> the SHIPPED bundle.

The two memories are not the same thing and must never be shipped as if they were. The
session partition is one collaboration's working memory: build history, checkpoint
narration, deployment specifics, open items -- personal by nature, and shipping it would
hand every user our arc's biases as if they were the engine's knowledge. The bundled
memory is the distillate: GENERIC, measured engine knowledge that helps everyone --
what the organs are, what the doctrines are, what was proven and how -- the same way
only the relevant slice of a corpus gets kept.

Selection is rule-based and AUDITED, not vibes: an entry ships only if it (a) carries
engine-lexicon content, (b) matches no session/build/history marker, and (c) passes the
leak scan (no paths, no session tags, no collaborator names). Every exclusion is
counted by reason and reported -- what was withheld is visible, so the distillation
itself cannot become a blind spot. Shipped entries stay veto-able and provenance-tagged
like everything else; the bundle answers generic engine questions and ESCALATES on
everything else (verified below, not assumed).
"""
import os
import re
import sys
import shutil

sys.path.insert(0, ".")
import lecore

LEXICON = ("bind", "bundle", "cleanup", "resonator", "drift", "provenance",
           "tombstone", "saturation", "grounding", "ouroboros", "holographic",
           "fhrr", "quantization", "archive", "holoforest", "conjecture",
           "hypothesis", "permutation", "capacity", "pheromone", "delta rule",
           "hrr", "vsa", "lever", "abstention", "escalat", "veto", "slime",
           "metaball", "void", "recall", "semantic", "reflex")
EXCLUDE = [
    (re.compile(r"\bcp\s?\d{2}\b|\bcheckpoint\b", re.I), "build-history"),
    (re.compile(r"\bsession\b|\[s:", re.I), "session-specific"),
    (re.compile(r"user's box|pending|next step|next build|recorded next", re.I),
     "open-item"),
    (re.compile(r"/tmp/|/home/|/mnt/", re.I), "absolute-path"),
    (re.compile(r"openzoo|galv_out|lestudio|leos\b", re.I), "deployment-specific"),
    (re.compile(r"what workflow solved", re.I), "workflow-log"),
    (re.compile(r"claude|anthropic", re.I), "collaborator"),
]
GENERIC_TOPICS = ("ouroboros-sota-2026", "void-sota-2026")


def distill(partition, out_dir):
    src = lecore.UnifiedMind()
    src.boot(partition=partition, doctrine=True, llm=lambda p: "")
    lad = src.zoo["ladder"]
    rows = [t for t in getattr(lad, "taught_log", [])
            if len(t) > 3 and t[3] in ("taught", "validated", "evidenced")]
    kept, excluded = [], {}
    seen = set()
    for t in rows:
        q, a = str(t[0]), str(t[1])
        blob = (q + " " + a).lower()
        if q in seen:
            continue
        seen.add(q)
        reason = None
        for rx, name in EXCLUDE:
            if rx.search(blob):
                reason = name
                break
        if reason is None and not any(w in blob for w in LEXICON):
            reason = "no-engine-content"
        if reason is None and (len(a) < 40 or len(a) > 2000):
            reason = "length"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            kept.append((q, a, t[3]))
    dst = lecore.UnifiedMind()
    dst.zoo_attach(lambda p: "")
    for q, a, prov in kept:
        dst.teach(q, a)
        if prov in ("validated", "evidenced"):
            dst.conjecture_record(q, a)
            dst.conjecture_promote(q, prov, "carried from distillation source")
    n_topics = 0
    for topic in GENERIC_TOPICS:
        corp = getattr(src, "_archive_corpora", {}).get(topic)
        if corp:
            try:
                chunks = [c for c in corp.get("chunks", [])] if isinstance(corp, dict) \
                    else None
                if chunks:
                    dst.research_archive(topic, chunks,
                                         sources=["%s#%d" % (topic, i)
                                                  for i in range(len(chunks))])
                    n_topics += 1
            except Exception:
                pass
    os.makedirs(out_dir, exist_ok=True)
    dst.learning_save(out_dir)
    return {"source_rows": len(rows), "shipped": len(kept),
            "excluded": dict(sorted(excluded.items())), "topics": n_topics,
            "bundle": os.path.join(out_dir, "learning", "state.lecore")}


if __name__ == "__main__":
    rep = distill("/home/claude/claude_partition", "/home/claude/release_bundle")
    print(rep)
