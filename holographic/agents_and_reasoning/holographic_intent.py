"""VSA-native question routing -- understand what is being ASKED from a blend of the question's
word meanings, instead of brittle regex templates.

WHY THIS EXISTS
---------------
answer()'s regex templates are exact but brittle: natural or verbose phrasing
("could you tell me whether a dog is an animal", "do you happen to know the capital of japan")
misses the template and abstains, even though the brain knows the answer. The fix is the engine's
own machinery: the text encoder already turns a string into a BUNDLE of its word meanings (the VSA
blend), and that bundle is a good INTENT signal -- questions of the same kind share their function
words, so they cluster. So this routes a question by encoding it (mind.perceive) and matching it to
per-intent prototypes (each the mean bundle of several example phrasings), then dispatches to the
brain's real operations.

THE ORDER PROBLEM, AND WHY THE BLEND IS ONLY HALF THE ANSWER. Bundling is COMMUTATIVE, so
"is a dog an animal" and "is an animal a dog" blend to nearly the same vector -- the blend can tell
WHAT KIND of question it is, but not WHICH concept is the subject and which is the object. So intent
comes from the blend, but the ARGUMENTS come from a concept-scan: find the words the mind actually
knows (its class labels + lexicon), and use their ORDER (first found = subject, last = ancestor) to
assign roles -- the direction the commutative bundle cannot supply. Intent-by-blend +
arguments-by-order is the whole idea, and each half does the job the other can't.

MEASURED (honest picture)
  * On natural/verbose phrasings the regex router abstained on (0/8), VSA intent routing got 7/8;
    the full pipeline answered 6/7 end to end -- including the DIRECTION case ("is an animal a dog"
    -> "No"), which the order-scan resolves and the blend alone cannot.
  * KEPT NEGATIVES: (1) very short questions with overlapping content words can confuse adjacent
    intents ("what's a salmon" leans IS_A because "a salmon" appears in IS_A examples); when an IS_A
    question names only one known concept the router ABSTAINS rather than guess (describing the lone
    concept could answer the wrong thing if the real subject is the unknown one). (2) Arguments must
    be concepts the mind KNOWS; an unknown concept yields no pair and the router abstains (correctly
    -- it does not fabricate). (3) classify/recall need an explicit text PAYLOAD, not a concept, so
    they stay with the templates; this router covers the relational intents (is_a / role / define /
    similar). (4) The intent prototypes are built from a fixed example set, so coverage is only as
    broad as those phrasings -- a heavily padded wording can fall below the intent floor and abstain.

This is a FALLBACK: answer() tries the exact templates first (fast, precise, backward-compatible)
and only calls this when they miss, so nothing that worked before changes. Pure NumPy; the intent
prototypes are cached on the mind (built once from its own encoder, so the atoms match the queries).
"""
import re
import numpy as np

# Example phrasings per relational intent -- varied on purpose so the prototype is the COMMON
# (mostly function-word) signal, not any one wording. These are the router's whole training set.
INTENT_EXAMPLES = {
    "IS_A": ["is a dog an animal", "is a cat a mammal", "are dogs animals", "is a bird an animal",
             "is a salmon a fish", "is an oak a plant", "could you tell me whether a dog is an animal",
             "is a wolf really a mammal", "do dogs count as animals", "is a cat a kind of animal"],
    "ROLE": ["what is the capital of france", "capital of japan", "what is the population of egypt",
             "what is the author of this book", "do you know the capital of spain",
             "what is the capital of italy", "tell me the capital of germany",
             "what is the color of the sky", "what is the area of canada"],
    "DEFINE": ["what is a dog", "define wolf", "what is an oak", "what does mammal mean",
               "describe a cat", "tell me about a salmon", "what exactly is a bird",
               "what is a fish", "explain what a tree is", "what is a mammal"],
    "SIMILAR": ["what is like a dog", "what is similar to a wolf", "what resembles a cat",
                "what is most like a fish", "what is comparable to an oak", "what is similar to a bird"],
    "CLASSIFY": ["what kind of text is this", "classify this sentence", "what genre is this passage",
                 "what type of writing is this", "what category does this text belong to"],
}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _router(mind):
    """Build (and cache on the mind) the intent prototypes: per intent, the mean bundle of its
    example phrasings, encoded with THIS mind's text encoder so the atoms match a query's."""
    if getattr(mind, "_intent_proto", None) is None:
        labels = sorted(INTENT_EXAMPLES)
        P = []
        for lab in labels:
            V = np.stack([np.asarray(mind.perceive(q, "text"), float) for q in INTENT_EXAMPLES[lab]])
            P.append(_unit(V.mean(0)))
        mind._intent_proto = (labels, np.stack(P))
    return mind._intent_proto


def route_intent(mind, question):
    """Classify a question's intent from the blend of its word meanings. Returns (intent, score)."""
    labels, P = _router(mind)
    sims = P @ _unit(np.asarray(mind.perceive(question, "text"), float))
    j = int(np.argmax(sims))
    return labels[j], float(sims[j])


def _define_struct(mind, word):
    near = mind.define(word, 5) if hasattr(mind, "define") else []
    chain = mind.climb(word)[0] if hasattr(mind, "climb") else [word]
    if near or len(chain) > 1:
        return {"kind": "define", "word": word, "via": "vsa",
                "meaning": [(w, round(float(s), 3)) for w, s in near], "is_a_chain": chain}
    return None


def route_question(mind, question, intent_floor=0.25):
    """Route a (possibly natural-language) question to the brain's real operation and return an
    answer()-style struct (so realize_answer and the abstention floors apply unchanged), or None to
    let the caller abstain. Intent from the blend; arguments from a known-concept scan with ORDER."""
    intent, score = route_intent(mind, question)
    if score < intent_floor:
        return None
    words = re.findall(r"[a-z']+", (question or "").lower())
    concepts = set(mind._class_labels()) | set(getattr(mind, "_lexicon_words", set()))
    found = [w for w in words if w in concepts]                  # known concepts, in question order
    roles = set(getattr(mind, "_fillers", {})) - {"is_a"}        # askable roles (e.g. capital)

    # ROLE: a known role word + a known concept -> read that role off the concept
    if intent == "ROLE":
        rf = [w for w in words if w in roles]
        if rf and found and found[-1] in mind._class_labels():
            val, conf = mind.read_role(found[-1], rf[0])
            if val is not None:
                return {"kind": "role", "concept": found[-1], "role": rf[0], "value": val,
                        "confidence": round(float(conf), 3), "via": "vsa"}

    # IS_A with two known concepts -> ORDER assigns subject (first) and ancestor (last)
    if intent == "IS_A" and len(found) >= 2:
        subj, anc = found[0], found[-1]
        reached, hops, tp = mind.is_a(subj, anc)
        return {"kind": "is_a", "subject": subj, "ancestor": anc, "answer": bool(reached),
                "hops": hops, "throughput": round(float(tp), 3), "chain": mind.climb(subj)[0], "via": "vsa"}

    # DEFINE / SIMILAR -> describe the named concept (its class + nearest learned meanings).
    # Restricted to these intents on purpose: describing the lone KNOWN concept of an IS_A question
    # whose SUBJECT is unknown ("is a dragon an animal" -- animal known, dragon not) would answer the
    # WRONG thing, so those abstain instead. The cost is that a short misrouted question ("what's a
    # salmon" leaning IS_A) abstains rather than being salvaged -- an honest, safe trade.
    if intent in ("DEFINE", "SIMILAR") and found:
        s = _define_struct(mind, found[0])
        if s is not None:
            return s
    return None
