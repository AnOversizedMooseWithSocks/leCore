"""holographic_catalog_p08.py -- catalog part 08: THE CODE-TOOLS ROBUSTNESS CARDS.

Provenance: the code/text tooling sweep. A find_capability audit for structural
codebase mapping, diagrams, spec conformance, document outlining, arbitrary-root
reference docs, and sandboxed execution returned only fallbacks -- the absence
that licensed holographic_repograph + holographic_docforge and the six faculties
carded here. Aliases below were written from a STRANGER's mouth (five phrasings
brainstormed per card before finalizing) and each card passed the 5/5
discoverability battery at close-out.
"""


def register_p08(c):
    c.register_capability(
        "repo_map",
        "Map a MIXED-LANGUAGE codebase (python/javascript/c): symbols per file, "
        "the file dependency graph from def/ref matching, deterministic "
        "PageRank ranking of which files matter, and a budgeted text skeleton "
        "(aider-style) -- large trees summarized without dumping them. "
        "focus= biases the rank toward files being worked on (personalized "
        "PageRank, the 50x aider convention). Truncation is always "
        "announced, never silent.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.repo_map('holographic/io_and_interop', "
                "budget_lines=30)['files'])",
        native=True,
        aliases=("map out a large codebase", "repo map", "what files matter most",
                 "summarize this codebase", "codebase overview",
                 "scan a javascript project", "index a c codebase",
                 "which files depend on which", "rank files by importance"),
        module="holographic_repograph")
    c.register_capability(
        "codebase_diagram",
        "Draw a codebase as DIAGRAM TEXT: mermaid flowchart or graphviz dot of "
        "the top-ranked files with reference-weighted dependency edges. Text "
        "diagrams diff, version, and render anywhere; drawing caps are "
        "announced in a note node.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.codebase_diagram('holographic/io_and_interop', "
                "fmt='mermaid', max_nodes=5).split(chr(10))[0])",
        native=True,
        aliases=("generate a codebase diagram", "draw the dependency graph",
                 "visualize my project structure", "mermaid diagram of code",
                 "architecture diagram from source", "graphviz of the repo"),
        module="holographic_repograph")
    c.register_capability(
        "spec_conformance",
        "Check a spec/SOP against a source tree WITHOUT hallucination: atomic "
        "claims, mechanical file:line evidence (every citation re-read from "
        "disk before it may appear), verdicts supported/partial/violated/"
        "unverifiable. Ungrippable prose is abstained on, never judged; an "
        "optional llm may propose finer claims but can never confirm one.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.spec_conformance('Provides `SpecChecker`.', "
                "'holographic/io_and_interop')['report'][0]['verdict'])",
        native=True,
        aliases=("check codebase against a spec", "does the code follow the sop",
                 "verify requirements are implemented", "compliance check on code",
                 "audit code against requirements", "spec coverage report"),
        module="holographic_repograph")
    c.register_capability(
        "document_outline",
        "Break a large text document into an ORGANIZED document with a table "
        "of contents: headed markdown keeps the author's structure; unheaded "
        "prose is cut at lexical-cohesion dips (TextTiling-style, "
        "deterministic). Content is reorganized, never rewritten; an optional "
        "llm may only rename section titles.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.document_outline('# A' + chr(10) + 'x' + chr(10) + "
                "chr(10) + '# B' + chr(10) + 'y')['toc'])",
        native=True,
        aliases=("break a document into sections", "table of contents",
                 "organize my research notes", "outline a long text",
                 "structure this document", "split notes into chapters"),
        module="holographic_docforge")
    c.register_capability(
        "docs_generate",
        "Generate a deterministic markdown REFERENCE for any python/js/c tree: "
        "every file, every definition with signature and line number, plus the "
        "author's own first docstring sentence where present -- docgen "
        "generalized from leCore's tree to arbitrary roots.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.docs_generate('holographic/io_and_interop')['defs'])",
        native=True,
        aliases=("generate documentation from code", "build a reference manual",
                 "document this codebase", "api reference from source",
                 "make docs for my project"),
        module="holographic_docforge")
    c.register_capability(
        "sandbox_run",
        "Run code in a SANDBOX: throttled child process with cpu/memory/"
        "file-size rlimits, scrubbed deterministic env (PYTHONHASHSEED=0), "
        "temp cwd, wall timeout, and loudly-marked output caps. Python always "
        "works; node/cc are used when installed and refused honestly when "
        "not.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sandbox_run('print(6*7)')['stdout'].strip())",
        native=True,
        aliases=("run code in a sandbox", "execute python safely",
                 "run untrusted code", "test a snippet with limits",
                 "safe code execution", "run this script isolated"),
        module="holographic_docforge")
    c.register_capability(
        "sop_run",
        "FOLLOW ORDERS: execute an authored text SOP through the mind. A model "
        "writes the plan (## step: / invoke: / python: / shell: / verify: / "
        "on_fail: / guidance:); the substrate runs it -- invoking faculties, "
        "sandboxing code, verifying every step -- and consults the model ONLY "
        "at declared guidance/escalation points. A scriptable SOP runs with "
        "ZERO model calls (llm_calls in the result proves it); an SOP that "
        "does not fully parse is refused before step 1.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sop_run('## step: a' + chr(10) + 'python: print(6*7)' "
                "+ chr(10) + 'verify: \"42\" in result[\"stdout\"]')['ok'])",
        native=True,
        aliases=("run an sop written by the model", "follow a step by step plan",
                 "execute orders from an llm", "run a standard operating procedure",
                 "let the llm plan and lecore execute", "run a scripted workflow",
                 "carry out a checklist automatically"),
        module="holographic_soprunner")
    c.register_capability(
        "sop_check",
        "Validate an authored SOP WITHOUT running anything: every problem "
        "named by line number, or ok with the step count. The author's "
        "(usually a model's) edit-until-clean loop before sop_run -- an order "
        "leCore cannot fully parse is an order it will refuse to follow.",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sop_check('## step: a' + chr(10) + 'python: 1+1'))",
        native=True,
        aliases=("validate a plan before running it", "check my sop syntax",
                 "lint a procedure", "will this plan run",
                 "dry run a workflow", "verify sop format"),
        module="holographic_soprunner")
    c.register_capability(
        "sop_save",
        "Save a NAMED SOP for later sop_run(name) -- the leOS macro_registry "
        "pattern on the durable KnowledgeStore, so procedures survive process "
        "restarts. Validates first (a plan that does not parse is refused, "
        "not stored); revisions are appends and the last save wins. Fetch "
        "with sop_load(name).",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(m.sop_save('demo', '## step: a' + chr(10) + 'python: 1'))",
        native=True,
        aliases=("save a named procedure for later", "store a reusable workflow",
                 "remember this sop", "keep a playbook across sessions",
                 "define a macro procedure", "load a saved sop"),
        module="holographic_soprunner")
    c.register_capability(
        "ask_chain",
        "MULTI-HOP CHAIN over the mind's own memory: ask_chain('tokyo', "
        "('capital','currency'), ('currency','language')) walks filler -> "
        "record -> filler hop by hop. RESTORED in sweep 63: this faculty was "
        "silently shadowed by the one-call ask() for its whole life -- the "
        "duplicate-name collision the split test guards.",
        example="import lecore; m=lecore.UnifiedMind(dim=512, seed=0); "
                "m.absorb([({'a':'x','b':'y'}, 'r1')]); "
                "print(m.ask_chain('x', ('a','b')))",
        native=True,
        aliases=("chain a question over memory", "multi hop question",
                 "follow relations hop by hop", "walk from value to value",
                 "answer through intermediate records"),
        module="holographic_unified_p02_fit_deterministic")
    c.register_capability(
        "explain_similarity",
        "WHY are two things similar: per-role comparison of two absorbed "
        "records -- for each shared role, both fillers and whether they "
        "match, with confidences. Not a bare cosine: the ROLES carry the "
        "explanation. RESTORED in sweep 63 from under explain(topic)'s "
        "shadow.",
        example="import lecore; m=lecore.UnifiedMind(dim=512, seed=0); "
                "m.absorb([({'a':'x'}, 'r1'), ({'a':'x'}, 'r2')]); "
                "print(m.explain_similarity('r1', 'r2')[0][3])",
        native=True,
        aliases=("why are these two similar", "compare two records role by role",
                 "what do these share", "per attribute comparison",
                 "explain the similarity"),
        module="holographic_unified_p02_fit_deterministic")
    c.register_capability(
        "boot",
        "BOOT THE SUBSTRATE LIKE FIRMWARE (cp34): POST -- measured self-checks "
        "with pass/fail per subsystem -- then mount memory and report readiness "
        "in one call. The in-process sibling of autoboot/agent_boot: use this "
        "when the mind already exists and you want its power-on self test. "
        "Carded explicitly in sweep 63: the docstring-derived card was "
        "alias-less and did not surface for its own name (the buried-audit "
        "dark-capability regression).",
        example="import lecore; m=lecore.UnifiedMind(dim=256, seed=0); "
                "print(type(m.boot()))",
        native=True,
        # NO "start" phrasings here (sweep 65): they displaced the shipped
        # autoboot alias "how do I start" from its own top-1 -- a new card's
        # battery must also prove it did not STEAL existing aliases.
        aliases=("boot the substrate", "power on self test", "post checks",
                 "run startup self checks", "substrate self check",
                 "is the engine healthy on boot"),
        module="holographic_unified_p20_zoo")
    register_p08_digest(c)
    return 13


def register_p08_digest(c):
    """The document-digest card (learning augmentation) -- appended by the
    digest sweep; called from register_p08 below via the module tail hook."""
    c.register_capability(
        "Document digest (learning augmentation, zero model calls)",
        "One call digests a large text FOR LEARNING: authored TOC (# and "
        "rule+TITLE dialects), kept-negative index (citations; the text stays "
        "in its section), per-section tf*idf signatures. Budgeted markdown "
        "render: negatives funded first, all truncation declared. Runs "
        "AUTOMATICALLY at ingestion -- KnowledgeStore.add files the companion "
        "note for any document >= DIGEST_THRESHOLD; original chunks stay "
        "byte-identical (augment, never edit). mind.document_digest(text).",
        example="m.document_digest('# A\n\nKEPT NEGATIVE: x.\n')['stats']",
        aliases=("digest a big document", "organize a large text for learning",
                 "index the kept negatives", "structure a document with no llm",
                 "auto digest on ingest"),
        native=True,
    )


_PART = "holographic_catalog_p08"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p08)
    return {"part": _PART, "cards": n}


if __name__ == "__main__":
    print(_selftest())
