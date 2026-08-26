"""Grounded answering -- construct a SHORT, COHERENT, ACCURATE sentence from what the mind
actually knows, instead of free-running the Markov generator (which, measured, is locally
fluent but globally incoherent) or dumping a raw struct.

WHY THIS EXISTS
---------------
The honest finding from the text-generation review: this engine's RELATIONAL layer answers
questions correctly and traceably (is_a chains, role lookups, learned-meaning similarity,
classification), while its GENERATIVE layer produces word-salad-with-local-fluency and cannot
cohere a sentence (its own docstrings say the missing piece is a high-capacity P(next|context)
this substrate does not have). So the right way to "answer a question with a sentence that makes
sense" is NOT to generate -- it is to RETRIEVE the facts (delegating entirely to
UnifiedMind.answer(), which routes a question to the brain's real operations) and then REALIZE
those facts into a short sentence.

The three properties this is built to hold:
  * ACCURATE -- the content is retrieved from the mind's own knowledge, never invented. When the
    mind does not know (unknown concept, low-confidence recall/classify, or the question falls
    through to the generation path), this ABSTAINS with an honest "I don't know" rather than
    fabricating -- the engine's calibrated-abstention discipline applied to language.
  * CONTEXTUAL -- the surface form matches the question shape (a yes/no question gets a yes/no
    with its justification; a role question gets the value; a what-is question gets the class).
  * NOT VERBATIM -- the sentence is CONSTRUCTED from the retrieved structure (the is_a chain, the
    role value, the learned-meaning neighbours), so it is a new sentence, not a copied source
    line. Verbatim recall happens only where the answer simply IS a stored value (a capital, a
    parent class) -- i.e. only when that is what was asked for.

This is template-based surface realization (the standard pre-neural NLG move) over a holographic
knowledge base -- deliberately using the parts of the engine that WORK (relational retrieval +
learned distributed meaning + calibrated abstention) and deliberately NOT the part that does not
(the free n-gram walk). Pure Python; no new learning; delegates retrieval to the mind.
"""


def _art(word):
    """'a' / 'an' by the next word's first sound (vowel-letter heuristic -- good enough for
    readable output; not a pronunciation dictionary)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _english_list(items):
    """['a','b','c'] -> 'a, b and c' (an Oxford-comma-free natural list)."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _chain_phrase(subject, chain, ancestor):
    """Render the taxonomic path subject -> ... -> ancestor as a clause, e.g.
    'a dog is a mammal, which is an animal' (1, 2, or 3+ links handled gracefully)."""
    path = chain[:chain.index(ancestor) + 1] if ancestor in chain else [subject, ancestor]
    mids = path[1:]                                   # the classes above the subject
    if not mids:
        return f"{_art(subject)} {subject} is {_art(ancestor)} {ancestor}"
    if len(mids) == 1:
        return f"{_art(subject)} {subject} is {_art(mids[0])} {mids[0]}"
    if len(mids) == 2:
        return (f"{_art(subject)} {subject} is {_art(mids[0])} {mids[0]}, "
                f"which is {_art(mids[1])} {mids[1]}")
    return (f"{_art(subject)} {subject} is {_art(mids[0])} {mids[0]}, "
            f"and ultimately {_art(mids[-1])} {mids[-1]}")


# confidence/score floors below which an answer is treated as "not really known" -> abstain
_RECALL_FLOOR = 0.30
_CLASSIFY_FLOOR = 0.30
_ROLE_FLOOR = 0.20
_SIM_FLOOR = 0.15


def realize_answer(result):
    """Turn a UnifiedMind.answer() result struct into a short, constructed sentence (or an honest
    abstention). `result` is exactly what mind.answer(question) returns. Returns a string."""
    if not isinstance(result, dict):
        return "I don't have an answer to that."
    kind = result.get("kind")

    # -- taxonomic yes/no, justified by the actual chain -----------------------
    if kind == "is_a":
        subj, anc = result["subject"], result["ancestor"]
        chain = result.get("chain", []) or [subj]
        known = len(chain) > 1                          # the subject has a known parent class
        if result.get("answer"):
            return f"Yes -- {_chain_phrase(subj, chain, anc)}."
        if not known:
            return (f"I don't have {subj} in my knowledge, so I can't say whether it's "
                    f"{_art(anc)} {anc}.")
        # subject IS known, just not under the asked ancestor -- show what it IS, accurately
        return (f"No -- {_chain_phrase(subj, chain, chain[-1])}, "
                f"not {_art(anc)} {anc}.")

    # -- a role of a concept (a capital, an author, ...) -- verbatim value, by design ----
    if kind == "role":
        if result.get("confidence", 1.0) < _ROLE_FLOOR or result.get("value") is None:
            return f"I'm not sure about the {result.get('role','that')} of {result.get('concept','it')}."
        return f"The {result['role']} of {result['concept']} is {result['value']}."

    # -- what is X: the class (from is_a) and/or what it's most like (learned meaning) ----
    if kind == "define":
        word = result["word"]
        chain = result.get("is_a_chain", []) or [word]
        near = [w for w, s in result.get("meaning", []) if s > _SIM_FLOOR][:3]
        parts = []
        if len(chain) > 1:
            if len(chain) == 2:
                parts.append(f"{_art(word).capitalize()} {word} is {_art(chain[1])} {chain[1]}.")
            else:
                parts.append(f"{_art(word).capitalize()} {word} is {_art(chain[1])} {chain[1]} -- "
                             f"more broadly, {_art(chain[-1])} {chain[-1]}.")
        if near:
            lead = "It's" if parts else f"{word.capitalize()} is"
            parts.append(f"{lead} closely related to {_english_list(near)}.")
        if parts:
            return " ".join(parts)
        return f"I know the word {word}, but I don't have enough about it to describe it."

    # -- nearest individual memory --------------------------------------------
    if kind == "recall":
        if result.get("score", 0.0) < _RECALL_FLOOR or not result.get("label"):
            return "I don't have anything like that in memory."
        return f"The closest thing I know of is {result['label']}."

    # -- classification of a piece of text ------------------------------------
    if kind == "classify":
        if result.get("score", 0.0) < _CLASSIFY_FLOOR or not result.get("label"):
            return "I'm not sure what kind of text that is."
        return f"That looks like {result['label']}."

    # -- the honest non-answers: question fell through to generation, or unmapped ----
    if kind in ("completion", "unknown", "none"):
        return ("I can't answer that from what I know -- I'm not a language model, but I can "
                "answer factual and relational questions (e.g. 'what is a dog?', 'is a dog an "
                "animal?', 'what is the capital of france?', 'define wolf').")

    return "I don't have an answer to that."
