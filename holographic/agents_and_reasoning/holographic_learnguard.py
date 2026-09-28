"""holographic_learnguard.py -- WHAT THE ENGINE MUST NEVER LEARN (sweep 179).

WHY THIS EXISTS
---------------
Moose: "We don't want to learn things like api keys, wallet seed phrases, or passwords. [...] I am also
concerned that we might learn static results from dynamic data. If asked to check the price of solana
currently, it's going to have a different answer almost every time you ask."

Measured before this module (fake values, scratch partition): teach() accepted an API key, a 12-word seed
phrase and a password; a FRESH mind loaded from the saved partition answered all three at T0. A taught
"$142.10" for "the current price of solana" and a resolve()d "72F and sunny" for "the weather right now"
came back at T0 too -- a stale reading served as a fact, forever. The sweep-178 tool-call records also
carried a tool's api_key in plain text in their argument preview.

TWO QUESTIONS, ANSWERED AT EVERY DOOR THAT LEARNS:
  1. SENSITIVE -- does the text carry a credential? (API key, private key, seed phrase, password, token.)
     Never learned, no override: the fix for a false positive is to rephrase, never to store a secret.
  2. VOLATILE  -- is this a READING of something that moves (a price, the weather, a balance, "right now")?
     Not learned as a fact. What SHOULD be learned for a volatile question is the ROUTE -- which tool to
     call -- and the tool-call learners (sweep 178) learn exactly that: labels are tool names, never values.
     A dated snapshot is a static fact ("SOL price as of 2026-09-22 15:00") and passes; so does a question
     with an explicit historical anchor. allow_volatile=True on teach() is the deliberate override.

THREE LAYERS, IN THIS ORDER (section numbers below):
  1-2. the PATTERN layer -- deterministic regex + one vendored wordlist. Strong shapes always refuse. Certain.
  3.   the TYPED layer (E4.2, only where the pattern layer is silent, only when a mind's SemanticGuard is passed):
       one typed decision -- credential / live_value / unclear / ordinary -- on the shared contrastive rule
       (holographic_protostore), built at runtime from shipped example lists (holographic_learnguard_examples),
       and it refuses only when the ANSWER has the shape of the thing. A guess: it refuses NEW learning visibly,
       with a correction path, and never deletes what is stored (load-time replay is pattern-only).

DESIGN RULES (the pattern layer)
  * Deterministic regex + one vendored wordlist (BIP-39 English, lecore_data/knowledge). No model, no guess.
  * STRONG patterns (a provider key prefix, a PEM block, a JWT, a 64-byte keypair array, a BIP-39 run,
    `password: <value>`) refuse on their own. WEAK shapes (64 hex digits, 80+ base58 chars, long mixed
    tokens) refuse only when a credential word is also present -- because sha256 digests, Solana
    signatures and transaction hashes have the same shapes and are PUBLIC. Bare "token" is NOT a
    credential word: this codebase is full of crypto tokens and VSA tokens.
  * A reason never echoes the secret. It names the rule so a false positive can be rephrased.
"""
import os
import re

# ---- 1. SENSITIVE -----------------------------------------------------------------------------------------

# Credential words: the CONTEXT that turns a weak shape into a secret, and a question into one asking for one.
_SECRET_CUE = re.compile(
    r"\b(pass(?:word|wd|phrase)s?|pwd|api[\s_-]?keys?|apikey|secret[\s_-]?keys?|client[\s_-]?secret|"
    r"private[\s_-]?keys?|priv[\s_-]?key|seed[\s_-]?phrases?|recovery[\s_-]?phrases?|mnemonic|keypair|"
    r"access[\s_-]?tokens?|auth[\s_-]?tokens?|bearer|refresh[\s_-]?tokens?|2fa|otp|one[\s-]time[\s-]code|"
    r"pin[\s_-]?(?:code|number)|credentials?|secrets?)\b", re.I)

# STRONG: these shapes are credentials whatever surrounds them.
_STRONG = [
    ("a PEM private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("an sk- style API key", re.compile(r"\bsk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}")),
    ("a Stripe-style secret key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("an AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("a GitLab / Hugging Face / npm token",
     re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}|\bhf_[A-Za-z0-9]{30,}|\bnpm_[A-Za-z0-9]{36}\b")),
    ("a JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # a Solana id.json keypair: a JSON array of exactly 64 byte values
    ("a 64-byte keypair array", re.compile(r"\[\s*(?:\d{1,3}\s*,\s*){63}\d{1,3}\s*\]")),
]

# `password: hunter2`, `"api_key": "..."`, `SECRET_KEY=...` -- a credential name ASSIGNED a value.
_ASSIGN = re.compile(
    # NOT bare "pass": calibration on the real partition (469 rows) found it matching test logs -- "PASS: ...",
    # "pass: lever_heal" -- 3 false positives, 3 of 3 flags. A credential assignment names the credential.
    r"\b(pass(?:word|wd|phrase)|pwd|api[_-]?key|apikey|secret[_-]?key|client[_-]?secret|private[_-]?key|"
    r"access[_-]?token|auth[_-]?token|refresh[_-]?token|seed[_-]?phrase|mnemonic)[\"']?\s*[:=]\s*[\"']?"
    r"([^\s\"',;}]{4,})", re.I)
# values that are obviously NOT a secret (documentation of a parameter, a placeholder, an env lookup)
_PLACEHOLDER = re.compile(r"^(none|null|true|false|\.\.\.|\*+|x{3,}|<.*|\$\{.*|os\.environ.*|getenv.*|"
                          r"your_.*|env\b.*|required|optional|string|str|bytes)$", re.I)

# WEAK: public things share these shapes (hashes, signatures, tx ids) -- a credential word must be present too.
_WEAK = [
    ("64 hex digits (a raw private key shape)", re.compile(r"\b(?:0x)?[0-9a-fA-F]{64}\b")),
    ("80+ base58 characters (a Solana secret-key shape)", re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{80,}\b")),
    ("a long mixed-character token", re.compile(r"(?<![\w/.])(?=[A-Za-z0-9+/_\-]*[A-Z])(?=[A-Za-z0-9+/_\-]*[a-z])"
                                                 r"(?=[A-Za-z0-9+/_\-]*\d)[A-Za-z0-9+/_\-]{32,}={0,2}")),
]

_BIP39 = None
SEED_RUN = 12            # the shortest standard seed phrase; 12 BIP-39 words in a row does not happen in prose
SEED_DISTINCT = 8        # ...of which at least this many are different words


def _bip39():
    """The BIP-39 English wordlist as a set (vendored, MIT; sha256 recorded in the knowledge manifest). An empty
    set if the file is missing -- the seed-phrase rule then goes quiet rather than guessing."""
    global _BIP39
    if _BIP39 is None:
        try:
            import lecore_data
            path = lecore_data.file("knowledge", "bip39_english.txt")
        except Exception:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "lecore_data", "knowledge", "bip39_english.txt")
        try:
            with open(path, encoding="utf-8") as f:
                _BIP39 = frozenset(w.strip() for w in f if w.strip())
        except OSError:
            _BIP39 = frozenset()
    return _BIP39


def _seed_run(text):
    """(start, end) character span of the first run of >= SEED_RUN consecutive BIP-39 words, or None."""
    words = _bip39()
    if not words:
        return None
    run_start, n = None, 0
    for m in re.finditer(r"[A-Za-z]+", str(text)):
        if m.group(0).lower() in words:
            if n == 0:
                run_start = m.start()
            n += 1
            # a run of one repeated word ("step step step ...") is not a seed: real phrases are nearly all
            # distinct words, so at least SEED_DISTINCT different words must appear in the run
            if n >= SEED_RUN and len({w.lower() for w in re.findall(r"[A-Za-z]+", str(text)[run_start:m.end()])}) >= SEED_DISTINCT:
                end = m.end()
                for m2 in re.finditer(r"[A-Za-z]+", str(text)[end:]):   # extend to the end of the run
                    if m2.group(0).lower() not in words:
                        break
                    end = end + m2.end()
                return run_start, end
        else:
            n = 0
    return None


# `my password is Hunter2-x9`, `the api key was abc123XYZ`, `pin code is 4821` -- a credential DISCLOSED IN A
# SENTENCE. Found by sweep 181's tests (a person types it this way, not as password=...): the assignment
# pattern above missed it and the wording would have been learned as a phrasing. The value must LOOK like a
# value (a digit, a symbol inside it, or mixed case; 4+ chars), so "my password is too short" and "the
# password is expired" stay ordinary sentences.
_DISCLOSE = re.compile(
    r"\b(pass(?:word|wd|phrase)|pwd|pin(?:[\s_-]?(?:code|number))?|api[\s_-]?key|secret[\s_-]?key|"
    r"private[\s_-]?key|access[\s_-]?token|auth[\s_-]?token|seed[\s_-]?phrase|passcode)\s+"
    r"(?:is|was|are|=|:)\s+[\"'`]?([^\s\"'`,;]{4,})", re.I)


# MORE WAYS PEOPLE PASTE A PASSWORD (found 2026-09-27 by the learning-loop audit, tools/audit_learning_loop.py
# check_guard_phrasings: the pattern layer caught only 4 of 10 ordinary wordings, and in the first audit run the
# missed ones reached the saved decisions and query log). Same value rule as _DISCLOSE -- the value must LOOK like
# a value -- so "password reset", "log in with google" and "that is my password manager" stay ordinary.
_SECRET_WORDS = (r"pass(?:word|wd|phrase)|pwd|pw|pword|pin(?:[\s_-]?(?:code|number))?|passcode|"
                 r"api[\s_-]?key|secret[\s_-]?key|private[\s_-]?key|access[\s_-]?token|auth[\s_-]?token")
_VALUE = r"[\"'`]?([^\s\"'`,;]{4,})"
_DISCLOSE_MORE = [
    # "the pw is X" / "pword was X" -- the short spellings _DISCLOSE did not list
    (re.compile(r"\b(pw|pword)\s+(?:is|was|=|:)\s+" + _VALUE, re.I), False),
    # "password X" / "use password X to log in" / "passphrase X" -- no verb at all
    (re.compile(r"\b(pass(?:word|wd|phrase)|pwd|passcode)\s+" + _VALUE, re.I), False),
    # "here is my password, X" / "password - X"
    (re.compile(r"\b(" + _SECRET_WORDS + r")\s*(?:,|-|\u2013|\u2014)\s*" + _VALUE, re.I), False),
    # "X is my password" / "X was the pin" -- the value FIRST
    (re.compile(r"(?:^|\s)" + _VALUE + r"\s+(?:is|was)\s+(?:my|the|our|your)\s+(?:" + _SECRET_WORDS + r")\b", re.I),
     False),
    # "login with X" / "sign in using X" -- stricter: a digit or an inner symbol is required, because mixed case
    # alone is a product name here ("log in with GitHub")
    (re.compile(r"\b(?:log\s?in|login|logon|sign\s?in|signin)\s+(?:with|using)\s+" + _VALUE, re.I), True),
]


def _looks_like_value(v, strict=False):
    """A pasted secret has a digit, a symbol inside it, or (unless strict) mixed case."""
    if re.search(r"\d", v) or re.search(r"\w[^\w\s]\w", v):
        return True
    return (not strict) and bool(re.search(r"[a-z]", v) and re.search(r"[A-Z]", v))


def _disclosure_hits(text):
    out = []
    for m in _DISCLOSE.finditer(str(text)):
        v = m.group(2).rstrip(".!?")
        if _PLACEHOLDER.match(v):
            continue
        if _looks_like_value(v):
            out.append(m)
    for rx, strict in _DISCLOSE_MORE:
        for m in rx.finditer(str(text)):
            v = m.group(m.lastindex).rstrip(".!?")
            if _PLACEHOLDER.match(v) or not _looks_like_value(v, strict):
                continue
            out.append(m)
    return out


def _assignment_hits(text):
    return [m for m in _ASSIGN.finditer(str(text)) if not _PLACEHOLDER.match(m.group(2))] + _disclosure_hits(text)


def sensitive_reason(*texts):
    """Why this text must never be learned, or None. Checks every text given (question AND answer): a strong
    shape anywhere refuses; a weak shape refuses only when a credential word appears in any of the texts."""
    joined = "\n".join(str(t) for t in texts if t is not None)
    for label, rx in _STRONG:
        if rx.search(joined):
            return "sensitive: contains %s -- secrets are never learned" % label
    if _seed_run(joined):
        return "sensitive: contains a %d+ word BIP-39 run (a wallet seed phrase) -- never learned" % SEED_RUN
    if _assignment_hits(joined):
        return "sensitive: a credential name is assigned a value (e.g. password: ...) -- never learned"
    if _SECRET_CUE.search(joined):
        for label, rx in _WEAK:
            if rx.search(joined):
                return "sensitive: %s next to a credential word -- never learned" % label
    return None


_ASKS_CREDENTIAL = re.compile(
    r"\b(pass(?:word|wd|phrase)s?|pwd|api[\s_-]?keys?|apikey|secret[\s_-]?keys?|client[\s_-]?secret|"
    r"private[\s_-]?keys?|priv[\s_-]?key|seed[\s_-]?phrases?|recovery[\s_-]?phrases?|mnemonic|keypair|"
    r"access[\s_-]?tokens?|auth[\s_-]?tokens?|bearer|refresh[\s_-]?tokens?|2fa|otp|one[\s-]time[\s-]code|"
    r"pin[\s_-]?(?:code|number))\b", re.I)
_ASKS_PASSWORD = re.compile(r"\b(pass(?:word|wd|phrase)|pwd|pin[\s_-]?(?:code|number))\b", re.I)
_ABOUT_PASSWORDS = re.compile(r"\bpass(?:word|phrase)s? (?:manager|policy|policies|reset|rules?|strength|"
                              r"requirements?|hash(?:ing)?|length)\b", re.I)


def _bare_credential(answer):
    """An answer that IS a value rather than an explanation: one whitespace-free token, 6+ chars, with a digit,
    a symbol or mixed case -- and not a path or URL (\"where is my keypair\" -> \"~/.config/solana/id.json\")."""
    a = str(answer).strip().strip("\"'`")
    if not a or any(c.isspace() for c in a) or len(a) < 6 or len(a) > 256:
        return False
    if "/" in a or "\\" in a or "://" in a:
        return False
    return bool(re.search(r"\d", a) or re.search(r"[^\w]", a) or (re.search(r"[a-z]", a) and re.search(r"[A-Z]", a)))


def sensitive_pair_reason(question, answer):
    """sensitive_reason over both texts, plus the PAIR rule: a question asking for a credential whose answer is
    a bare credential-shaped value (\"the admin password\" -> \"hunter2-Moose!\")."""
    r = sensitive_reason(question, answer)
    if r:
        return r
    # The PAIR rule uses the NARROW cue (specific credential nouns), not the broad one: tests/test_mcp_server's
    # release audit teaches "secret 0 of session 1" -> "payload-1-0" -- the loose word "secret" plus a
    # bare token was refused, a real false positive. Weak SHAPES still take the broad cue (they need a key shape).
    if _ASKS_CREDENTIAL.search(str(question)) and _bare_credential(answer):
        return "sensitive: the question asks for a credential and the answer is a bare value -- never learned"
    # A question that asks for a PASSWORD specifically gets no benefit of the doubt on shape: probed, an
    # all-lowercase passphrase ("correcthorsebatterystaple") passed the digit/symbol/mixed-case test above.
    # Talking ABOUT passwords (a manager, a policy, a reset) is not asking for one.
    if _ASKS_PASSWORD.search(str(question)) and not _ABOUT_PASSWORDS.search(str(question)):
        a_ = str(answer).strip().strip("\"'`")
        if a_ and not any(c.isspace() for c in a_) and len(a_) >= 6 and "/" not in a_ and "://" not in a_:
            return "sensitive: the question asks for a password and the answer is a bare value -- never learned"
    return None


# ---- 2. VOLATILE ------------------------------------------------------------------------------------------

# Words that make ANY measurement a reading of the present moment.
_LIVE = re.compile(r"\b(current(?:ly)?|right now|now|today|tonight|this (?:morning|afternoon|evening|week|month)|"
                   r"latest|live|at the moment|as of now|real[\s-]?time|up[\s-]?to[\s-]?date)\b", re.I)
# Quantities that move on their own, cue or no cue. Chosen against THIS codebase's vocabulary: no bare
# "volume" (render volumes), "temperature" (blackbody physics), "balance" after "white", "slot", "supply".
_MOVING = re.compile(r"\b(prices?|priced at|trading at|worth|market[\s-]?cap|mcap|24h volume|trading volume|"
                     r"liquidity|tvl|apy|apr|funding rate|(?<!white )balances?|weather|forecast|humidity|"
                     r"exchange rate|gas (?:fees?|price)|odds|share price|stock price|circulating supply|"
                     r"followers|holders)\b", re.I)
# "what is btc at", "where's sol trading", "whats eth at?" -- asking where a THING is AT (sweep 181, from the hand-
# written crypto questions): the moving-quantity words above are absent, the question still asks for a reading.
# Only the WHOLE question shape counts (a name, then at/trading, then the end), so "what is the tent at the
# fair" is not caught; a non-measurement answer ("the vet") is never volatile anyway.
_AT_READING = re.compile(r"^\s*(?:what|where|wheres|whats)(?:'s|s| is| are)?\s+\$?[a-z0-9.]{2,12}\s+"
                         r"(?:at|trading(?: at)?)\s*[?!.]*\s*$", re.I)
# An explicit anchor in time makes it a snapshot -- a static fact -- not a reading of now.
_ANCHOR = re.compile(r"\b(1[89]\d\d|20\d\d)\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}\b|"
                     r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\bat (?:launch|genesis|listing)\b|\ball[\s-]time\b|"
                     r"\bhistoric(?:al|ally)?\b", re.I)
# A question about a MECHANISM is not a reading, however many numbers its answer holds: "how is the pool price
# computed" (bench_learnguard: 10 of 10 wrongly refused before this rule). "How much" / "how many" are NOT
# here -- those ask for a value, and a value of a moving thing is exactly a reading.
_MECHANISM = re.compile(r"(why\b|how (?:is|are|does|do|did|can|could|should|would|to)\b|explain\b|"
                        r"what (?:does|do) .+ mean\b|what is the (?:formula|definition|meaning)\b)", re.I)
_MEASUREMENT = re.compile(r"\d|\b(sunny|cloudy|overcast|rain(?:y|ing)?|snow(?:y|ing)?|clear|storm(?:y)?|fog(?:gy)?|"
                          r"windy|drizzle)\b", re.I)
# Questions that are ROUTES or RECORDS, not facts: the tool reflex row ("which tool answers this shape") and a
# decision record written to memory ("which option was chosen"). Their answers are tool names / JSON, not
# readings -- so they are checked for secrets but never refused as volatile.
ROUTE_PREFIXES = ("toolreflex: ", "decision: ")
# How long an answer can be and still be a READING rather than an explanation. A moving quantity (price,
# weather) gets more room -- "SOL is trading at $142.10, up 3% on the day" -- than a bare live word, which
# is common in plain prose ("now", "latest") and so has to meet a terse answer to count.
MAX_READING_WORDS = 24
MAX_LIVE_READING_WORDS = 12


def volatile_reason(question, answer):
    """Why this (question, answer) is a reading of something that moves, or None. Volatile when the QUESTION
    carries a live cue or names a moving quantity, the ANSWER carries a measurement, and the question has no
    explicit time anchor. Route and record rows are exempt."""
    q, a = str(question or ""), str(answer or "")
    if q.startswith(ROUTE_PREFIXES):
        return None
    if _ANCHOR.search(q):
        return None
    if _MECHANISM.match(q.strip()):
        return None
    # A READING IS SHORT. Calibration on the real partition (469 rows): 19 of 19 volatile flags were dev-log
    # questions -- "what is unicron's install pipeline now", "where does the bundle live" -- whose answers
    # were 33-141 word EXPLANATIONS that happen to contain numbers. A price or a weather report is a handful
    # of words; an explanation of how prices are computed is not a reading and is fine to learn.
    n_words = len(a.split())
    cue = (_MOVING.search(q) if n_words <= MAX_READING_WORDS else None) or \
          (_AT_READING.search(q) if n_words <= MAX_READING_WORDS else None) or \
          (_LIVE.search(q) if n_words <= MAX_LIVE_READING_WORDS else None)
    if cue and _MEASUREMENT.search(a):
        return ("volatile: '%s' is a reading of something that changes -- learn the tool that fetches it, "
                "or teach a dated snapshot ('... as of 2026-09-22'), or pass allow_volatile=True" % cue.group(0))
    return None


# ---- 3. SEMANTIC -- the TYPED guard: what does the question MEAN? (sweep 180, rebuilt as E4.2) -------------------
#
# Moose: "In order for something to be learned the system needs to understand something semantically. The same
# goes for being able to filter something." The patterns above only know the phrasings they were written for:
# "sol?" -> "$142" and "what's the pw for the router" -> "Xk29!mQpa" sail past them, while memory would happily
# recall both later. So where the pattern layer is SILENT, the guard asks what the question means -- and since
# E4.2 it answers with ONE TYPED DECISION over four options:
#
#     credential  -- asks for (or hands over) a secret's VALUE: "the pin for my visa card", "save my anthropic key"
#     live_value  -- asks for a READING of something that moves: "how many yen for a dollar", "is the api up"
#     unclear     -- too bare to tell what it asks: a bare ticker ("ada?"), a bare referent ("the code?"). The
#                    owner's rule: a bare crypto ticker leads to CLARIFICATION, not action -- and not to a stored fact
#     ordinary    -- everything else, including TALK ABOUT secrets and prices: "I forgot my passcode", "how do I
#                    change my PIN", "how is TVL calculated", "what's my credit score", "what is 2+2"
#
# THE MODEL, derived at runtime from shipped EXAMPLE LISTS only (no dense weights are shipped):
#   features  guard_features(): character 3..5-grams (sub-word: "pw" near "password", typos survive) + words +
#             word bigrams (the FRAME: "what's my" vs "how do i") + the SHAPE (word count, a '?') so bare
#             questions resemble each other, IDF-weighted over the shipped examples, signed-hashed into 2048 dims.
#   rows      holographic_protostore.ProtoStore, the ONE contrastive rule every learning door shares: one unit
#             prototype per example group (a CLINC150 / Banking77 intent, a hand-written group), keyed by its type
#             so rows of the same type never push each other; one InfoNCE pass over the examples pulls each truth
#             row and pushes the rivals it was confused with -- which is how the HARD NEGATIVES (Banking77 PIN and
#             passcode talk, CLINC150 static-number intents, public identifiers) actually train something.
#   decision  per type, the best row cosine; a non-ordinary type wins only when it leads ORDINARY by more than
#             its TYPED_MARGIN, and credential / live_value also need a row cosine >= GUARD_FLOOR -- a type that
#             leads on less resemblance than that makes the question UNCLEAR (ask), not a guess (typed_value). All
#             chosen on the train half, see below. decide() returns the value, the ranked types, the leads and a
#             calibrated p_correct (DoorCalibrator; None until it has labels).
#   engine    the rows also hold a FROZEN snapshot of this engine's own catalog phrases as ordinary
#             (holographic_learnguard_examples.ENGINE_EXAMPLES): without it, engine talk answered with a tool name
#             ("smooth a bumpy mesh" -> "Voxelization") was refused on noise. Frozen, so a new catalog card changes
#             nothing -- the sweep-180 parity bug cannot come back.
#   learning  every pattern refusal and every learn_guard_example() adds an EXEMPLAR row (the question itself) and
#             runs one InfoNCE step, with the teacher's estimated flip rate (TeacherNoise.eps_hat, 0 until
#             re-labels exist) -- the noisy-label form of the same rule. Learned examples persist as TEXT with the
#             partition (state()/restore()) and are replayed into the rows, so nothing dense is ever stored.
#
# THE ANSWER STILL DECIDES (sweep 180's lesson, kept): a type alone never refuses. credential needs an answer that
# could be a secret (_secret_shaped, a PIN, a long lower-case passphrase -- not a method or card name); live_value a
# short number-led reading (_reading_shaped); unclear a bare measurement or a random-looking token (_bare_value).
# On the train half the answer gate is what makes a usable margin safe.
#
# MEASURED -- see TYPED_MARGIN below for the numbers, and docs/research/evidence/bench_learnguard.txt for the run.

import hashlib as _hashlib
import math as _math

# Sweep 180's hand-written examples: written alongside held-out set B. They stay, as three options of the typed
# store ("s180:credential" / "s180:reading" / "s180:normal"), so nothing the old guard knew is forgotten.
CREDENTIAL_EXAMPLES = (
    "what is my password", "the admin password", "my api key for the exchange", "what's the wifi password",
    "my wallet seed phrase", "the recovery phrase for my wallet", "my private key", "the login for the server",
    "what is the database password", "my github token", "the ssh key for the box", "what pin unlocks it",
    "the 2fa backup codes", "my phantom wallet secret", "credentials for the admin panel",
    "the secret key for signing", "my metamask words", "the passphrase for the vault", "what do I log in with",
    "remember my password", "save my api key", "the keypair for the deployer", "the token for the bot",
    "my exchange secret", "the root password")
READING_EXAMPLES = (
    "what is the price of solana", "sol price", "how much is bitcoin", "eth price right now", "what's the weather",
    "is it raining", "temperature outside", "what is my wallet balance", "how much sol do I have", "tvl of the pool",
    "current apy", "gas fees now", "what time is it", "what's the date today", "market cap of bonk",
    "24h volume on jupiter", "funding rate on sol perp", "how many holders does gweil have",
    "what is the score of the game", "odds for tonight", "how is the market doing", "what is sol trading at",
    "price of jup", "how much is my bag worth", "exchange rate usd to eur", "current block height",
    "how many followers do I have", "what's the forecast for tomorrow", "liquidity in the pool", "is the server up")
# Static facts whose answers are short numbers. KEPT NEGATIVE (sweep 180): written AFTER seeing three real
# partition false positives, so the partition's 0 was partly fitted -- the E4.2 held-out set is the clean number.
NORMAL_EXAMPLES = (
    "what is the boiling point of water", "how many moons does jupiter have", "what is the speed of light",
    "how tall is mount everest", "what is the tensile strength of steel", "how many bytes in a kilobyte",
    "what year was python released", "what is the atomic number of carbon", "how far is the moon",
    "what is the melting point of iron", "how many legs does a spider have", "what is pi to five digits",
    "what is the population of france", "how long is a marathon", "what is the density of aluminium",
    "how many days in a leap year", "what is the half life of carbon 14", "what port does ssh use",
    "how many bits in a byte", "what is the default dim of the engine", "what is the max supply of bitcoin",
    "what is the gravity on mars", "how many sides does a hexagon have", "what is absolute zero",
    "what is the resolution of 4k", "how many keys on a piano", "what is the wavelength of red light",
    "how old is the universe", "what is the radius of the earth", "how many tiles does the trace use",
    "what is the capacity of the bundle", "what is the size of the model file", "what was the sweep result",
    "how many tests passed", "what is the recall at ten", "how many rows are in the partition")
# Generic chat shorthand only. NOT tickers: mapping "sol" -> "solana price" was fitted to a held-out set (sweep 180).
SLANG = {"rn": "right now", "pw": "password", "pwd": "password", "creds": "credentials", "passcode": "password",
         "mcap": "market cap", "tvl": "total value locked", "atm": "at the moment", "rly": "really"}
MAX_LEARNED = 500                 # learned examples per kind; the oldest drop first (a bounded, auditable set)

TYPED_OPTIONS = ("credential", "live_value", "unclear", "ordinary")
# learn(q, kind) accepts the typed names and the sweep-180 names; the LEARNED lists keep the sweep-180 names so a
# partition written by either version restores into the other (old readers ignore the new "unclear" list).
_TYPE_OF_KIND = {"credential": "credential", "reading": "live_value", "live_value": "live_value",
                 "unclear": "unclear", "normal": "ordinary", "ordinary": "ordinary"}
_KIND_OF_TYPE = {"credential": "credential", "live_value": "reading", "unclear": "unclear", "ordinary": "normal"}

GUARD_DIM = 2048                  # hashed feature space (4096 measured no better on the train half)
GUARD_EPOCHS = 1                  # InfoNCE passes over the shipped examples at build time (3 and 5 measured no better)
# The softmax of each build step runs over the 32 nearest rows plus the truth (ProtoStore's documented cheap form for
# many options): with the 300-card engine snapshot the store has 364 rows, and the full softmax took the build from
# 2.9 s to 7.5 s per process; topk 32 brings it back to 3.2 s.
GUARD_TOPK = 32
# How far a type must LEAD the best ordinary row before the typed decision says so, and the FLOOR: the row cosine a
# credential / live_value decision needs at all (a question that resembles nothing the guard has seen is not refused
# on noise -- open-set abstention). Chosen on the TRAIN half only by tools/bench_learnguard.py --cv DATA: 5-fold CV
# through build_typed_store; per type the smallest margin with 0 typed false positives on (a) the train half's
# ordinary questions with REALISTIC answers and (b) the catalog's TUNING half, each alias answered by its card's
# does / name / method; + a safety; unclear floored at 0. The (safety, floor) kept: FALSE POSITIVES FIRST -- the
# lowest leave-one-fold-out estimate (margins picked without a fold, false positives counted on it), then the most
# catches. MEASURED (train half; caught credential / live / unclear of 79 / 286 / 29; train FP; leave-one-fold-out FP
# of 741):
#     safety 0.02  floor 0.00   62 / 247 / 27   0  1        safety 0.03  floor 0.00   60 / 244 / 27   0  0
#     safety 0.02  floor 0.18   65 / 250 / 27   0  1        safety 0.03  floor 0.18   63 / 250 / 27   0  0  <-- shipped
#     safety 0.04  floor 0.18   62 / 247 / 27   0  0        safety 0.03  floor 0.20   63 / 250 / 27   0  1
# HISTORY, kept loud (docs/research/evidence/bench_learnguard.txt has every run): the held-out set was judged THREE
# times. Freeze 1 (safety 0.02, no floor, no engine snapshot, looser answer gates) then broke two CI tests
# (test_route_tiered, test_mcp_server) and 5 tests/guides facts; freeze 2 added the snapshot, the floor and the gates
# that fixed those; freeze 3 changed only what an under-the-floor lead means (unclear, not ordinary). A guess must
# not block a fact; the recall this costs is reported, not bought.
TYPED_MARGIN = {"credential": 0.0456, "live_value": 0.0516, "unclear": 0.0}
GUARD_FLOOR = 0.18


def normalise_question(q):
    """Lower-case, keep word-ish tokens, expand generic shorthand (SLANG)."""
    return " ".join(SLANG.get(w, w) for w in re.findall(r"[a-z0-9/?]+", str(q).lower())).strip()


# Feature weights: characters, words, word bigrams, shape. MEASURED during design (tools/bench_learnguard.py --cv on
# the train half, before the engine snapshot and the floor; margins at 0 FP + 0.02; caught credential / live / unclear
# of 79 / 286 / 29):
#     char 1.0, word 1.0, bigram 0.5   66 / 244 / 23      word 2.0, bigram 1.0   65 / 249 / 21
#     char 1.0, word 2.0, bigram 0.5   67 / 247 / 21      word 3.0, bigram 1.5   64 / 248 / 18
#     char 0.5, word 1.0, bigram 0.5   65 / 249 / 27  <-- shipped: the best total; +-2 on 79 is noise, the unclear
#                                                         gain (the shape weighs more when characters weigh less) is not
GUARD_WEIGHTS = {"char": 0.5, "word": 1.0, "bigram": 0.5, "shape": 1.0}


def guard_features(text):
    """{feature: weight} -- what the typed guard reads in a question (see the section comment). MEASURED on the
    train half during design (an earlier harness, 5-fold CV, 0 FP on realistic answers, before the extra examples):
    characters only caught credential 61/79, live 213/286, unclear 1/29; + words, bigrams and shape: 63 / 214 / 26
    (the shape is what places a bare ticker); + IDF: 63 / 226 / 21; + 30 more dataset examples per intent: 63 / 245
    / 25; + the hand-written train-only extras: 68 / 245 / 26. KEPT NEGATIVES: dim 4096 (no better), one row per
    TYPE instead of per intent (55 / 175 / 17), nearest-neighbour exemplars instead of prototypes (53 / 198 / 9),
    prototype + exemplar mix (57 / 226 / 4), 3-5 InfoNCE epochs (no better), boundary exemplars (61 / 244 / 25)."""
    W = GUARD_WEIGHTS
    t = normalise_question(text)
    s = " " + t + " "
    f = {}
    for n in (3, 4, 5):
        for i in range(len(s) - n + 1):
            g = "c:" + s[i:i + n]
            f[g] = f.get(g, 0.0) + W["char"]
    words = t.split()
    for w in words:
        f["w:" + w] = f.get("w:" + w, 0.0) + W["word"]
    for a, b in zip(["<s>"] + words, words + ["</s>"]):
        k = "b:%s_%s" % (a, b)
        f[k] = f.get(k, 0.0) + W["bigram"]
    nw = len([w for w in words if w != "?"])
    f["s:nw%d" % min(nw, 5)] = W["shape"]
    if "?" in t:
        f["s:q"] = 0.5 * W["shape"]
    return f


class GuardEncoder:
    """IDF-weighted, signed feature hashing into GUARD_DIM dims, unit length. The IDF is fitted on the texts given
    (the shipped examples), so the encoder -- like everything else here -- is a pure function of the lists."""

    _SLOT = {}                                    # feature -> (index, sign); shared, tiny (ints only)

    def __init__(self, texts, dim=GUARD_DIM, features=None):
        df = {}
        for t in texts:
            for g in (features[t] if features is not None else guard_features(t)):
                df[g] = df.get(g, 0) + 1
        self._fit(df, len(texts), dim)

    def _fit(self, df, n_texts, dim):
        """The IDF from document frequencies -- the ONE formula, shared by __init__ and from_df (the disk cache)."""
        self.dim = int(dim)
        self.df, self.n_texts = df, int(n_texts)          # kept so the fit can be cached and refitted exactly
        n = float(n_texts)
        # smoothed IDF: a feature in every example weighs 1, a rare one up to log(n+1)+1; an unseen one counts as
        # rare (it is information the examples never had)
        self.idf = {g: _math.log((n + 1.0) / (c + 1.0)) + 1.0 for g, c in df.items()}
        self.unseen = _math.log(n + 1.0) + 1.0

    @classmethod
    def from_df(cls, df, n_texts, dim=GUARD_DIM):
        """An encoder refitted from saved document frequencies: the same IDF, bit for bit (same ints, same formula),
        without recomputing the features of every example (0.2 s per process -- cached_typed_store's loaded path)."""
        enc = cls.__new__(cls)
        enc._fit(df, n_texts, dim)
        return enc

    def _slot(self, g):
        s = self._SLOT.get((g, self.dim))
        if s is None:
            d = _hashlib.sha256(g.encode("utf-8")).digest()      # hashlib, never hash(): the same on every run
            s = self._SLOT[(g, self.dim)] = (int.from_bytes(d[:4], "big") % self.dim, 1.0 if d[4] & 1 else -1.0)
        return s

    def __call__(self, text, features=None):
        """The unit encoding of one question (a float64 row of length dim)."""
        import numpy as np
        f = features if features is not None else guard_features(text)
        # np.bincount sums each slot's contributions in INPUT ORDER starting from 0.0 -- exactly the old
        # `v[i] += sg * w * idf` loop in the features' order, so the bits are the same (backlog G2: the base rows'
        # sha256, the IDF and 500 decisions were compared before/after). With the slot cache read inline it replaces
        # ~110 numpy scalar writes and ~110 method calls per question. MEASURED on the 3,360 shipped examples, best of
        # 3 at load ~2.4: 0.39 s -> 0.28 s (the rest is the Python walk over the features themselves).
        slots, idf, unseen, dim = self._SLOT, self.idf, self.unseen, self.dim
        idx, val = [], []
        for g, w in f.items():
            s = slots.get((g, dim))
            if s is None:
                s = self._slot(g)
            idx.append(s[0])
            val.append(s[1] * w * idf.get(g, unseen))
        v = np.bincount(np.asarray(idx, dtype=np.intp), weights=np.asarray(val, dtype=np.float64),
                        minlength=dim) if idx else np.zeros(dim)
        n = float(np.sqrt(v @ v))
        return v / n if n > 0 else v


def shipped_examples():
    """[(text, type, option)] -- every example the base guard is built from: the E4.2 lists
    (holographic_learnguard_examples) and the sweep-180 lists above."""
    from holographic.agents_and_reasoning.holographic_learnguard_examples import examples
    out = list(examples())
    out += [(t, "credential", "s180:credential") for t in CREDENTIAL_EXAMPLES]
    out += [(t, "live_value", "s180:reading") for t in READING_EXAMPLES]
    out += [(t, "ordinary", "s180:normal") for t in NORMAL_EXAMPLES]
    return out


def _guard_encoder(examples, dim):
    """(encoder, {text: features}) for an example list -- the IDF fit. Features are computed once per text and shared
    by the IDF fit and the encoding."""
    F = {}
    for t, _, _ in examples:
        if t not in F:
            F[t] = guard_features(t)
    return GuardEncoder([t for t, _, _ in examples], dim=dim, features=F), F


def build_typed_store(examples, epochs=GUARD_EPOCHS, dim=GUARD_DIM, _encoder=None):
    """(encoder, ProtoStore) from [(text, type, option)]. Rows: one per option, at the unit mean of its examples,
    keyed by type. Then `epochs` InfoNCE passes in a sha256 order of the texts (deterministic, no RNG). The bench's
    cross-validation calls this with the held-in folds, so the tuned numbers come from THIS code path.
    (_encoder: an (encoder, features) pair already fitted on these examples -- cached_typed_store passes it.)"""
    import numpy as np
    from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
    enc, F = _encoder if _encoder is not None else _guard_encoder(examples, dim)
    X = {t: enc(t, F[t]) for t in F}
    # mine=False: the reverse-direction confusion miner (E1.2) is not read by this door, and its per-candidate loop
    # was a third of the build time (measured 0.7 s of 2.9 s)
    st = ProtoStore(dim, tau=0.05, lr=0.3, topk=GUARD_TOPK, name="learnguard", mine=False)
    groups = {}
    for t, typ, opt in examples:
        groups.setdefault(opt, (typ, []))[1].append(t)
    # ONE batch (backlog G2): add_option per group grew both matrices by one row each time -- 728 whole-matrix
    # copies, MEASURED 0.52 s of the 2.9 s build. add_options gives the same rows bit for bit with one growth.
    opts = sorted(groups)
    st.add_options(opts, [np.stack([X[t] for t in groups[o][1]]) for o in opts], [groups[o][0] for o in opts])
    order = sorted(examples, key=lambda e: _hashlib.sha256(("guard-epoch|" + e[0]).encode("utf-8")).hexdigest())
    for _ in range(int(epochs)):
        for t, _typ, opt in order:
            st.update(X[t], opt)
    return enc, st


# ---- the base build, cached on disk by CONTENT (backlog G2) -----------------------------------------------------
#
# WHY: the base rows are a pure function of the shipped example lists, the constants above and the code of this module
# and holographic_protostore -- and building them is 3,363 sequential InfoNCE steps over a 364 x 2048 store (each
# step reads the whole 6 MB prototype matrix: memory-bound, no bit-identical shortcut). MEASURED on the 2-core box:
# 2.9 s quiet, 26-39 s under 4x load, paid by the FIRST teach of EVERY process -- every pytest-xdist worker, every
# service start -- which pushed tests/test_meaning.py and tests/test_external_abstention.py over CI's 15 s budget.
# The vectorised encoder and the batched rows took the build to ~2.3 s; the rest is the rule itself.
# So the finished rows are cached in the temp dir (the catalog's lecore_fc_<hash>.json memo is the same pattern),
# keyed by sha256 of EVERYTHING the build reads: the examples, the constants, the SOURCE BYTES of this module and of
# holographic_protostore (any edit, even a comment, retires the file -- conservative on purpose), numpy's version, the
# Python version and the machine. A/P are stored in float64 through ProtoStore.state(dtype="float64"): the reload is
# BIT-EXACT (checked: the same A / P / labels / counts digests and the same decisions as a fresh build). The encoder
# (the IDF) is always refitted -- 0.2 s, and one less thing to trust. A missing, corrupt or mismatched file means a
# fresh build (and a rewrite); the write is atomic (temp name + os.replace), so pytest-xdist workers racing to write
# it cannot leave a half file. Nothing learned is ever cached here: learned exemplars live with the partition as text.
#   LECORE_GUARD_CACHE=0        always build (no read, no write)
#   LECORE_GUARD_CACHE_DIR=...  where the file lives (default: tempfile.gettempdir())
GUARD_CACHE_WAIT_S = 60.0          # how long a process waits for another's build before building itself (see below)
GUARD_CACHE_VERSION = 2                            # 2: the file also holds the encoder's document frequencies


def _guard_cache_key(examples, epochs, dim):
    """sha256 over everything the base build is a function of (see the section comment)."""
    import platform
    import sys
    import numpy as np
    from holographic.agents_and_reasoning import holographic_protostore as _ps
    h = _hashlib.sha256(("learnguard-base|v%d|%d|%d|%d|" % (GUARD_CACHE_VERSION, int(epochs), int(dim),
                                                            GUARD_TOPK)).encode("utf-8"))
    h.update(repr((sorted(GUARD_WEIGHTS.items()), sorted(SLANG.items()))).encode("utf-8"))
    for t, typ, opt in examples:
        h.update(("%s\x1f%s\x1f%s\x1e" % (t, typ, opt)).encode("utf-8"))
    for path in (__file__, _ps.__file__):
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(("|%s|%s|%s" % (np.__version__, sys.version.split()[0], platform.machine())).encode("utf-8"))
    return h.hexdigest()[:24]


def cached_typed_store(examples, epochs=GUARD_EPOCHS, dim=GUARD_DIM, cache_dir=None):
    """build_typed_store with the content-keyed disk cache described above -> (encoder, ProtoStore, how) where how is
    'loaded', 'built' or 'built (cache off)'. Same encoder, same rows, bit for bit, whichever path ran: the file holds
    A and P in float64 plus the encoder's document frequencies (the IDF is refitted from them with the same formula)."""
    import json
    import tempfile
    import numpy as np
    from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
    if os.environ.get("LECORE_GUARD_CACHE", "1").strip().lower() in ("0", "off", "false", "no"):
        enc, st = build_typed_store(examples, epochs=epochs, dim=dim)
        return enc, st, "built (cache off)"
    try:
        key = _guard_cache_key(examples, epochs, dim)
    except OSError:                                   # a source file cannot be read: no trustworthy key, just build
        enc, st = build_typed_store(examples, epochs=epochs, dim=dim)
        return enc, st, "built (cache off)"
    folder = cache_dir or os.environ.get("LECORE_GUARD_CACHE_DIR") or tempfile.gettempdir()
    path = os.path.join(folder, "lecore_guard_%s.npz" % key)

    def _load():
        """(encoder, store) from the file, or None when it is missing, corrupt or not this key's."""
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(bytes(z["meta"]).decode("utf-8"))
                A, P = np.array(z["A"]), np.array(z["P"])
                df_keys = json.loads(bytes(z["df_keys"]).decode("utf-8"))
                df_counts = [int(c) for c in z["df_counts"]]
            n = len(meta.get("labels", ()))
            if (meta.get("cache_key") != key or A.shape != (n, dim) or P.shape != (n, dim)
                    or len(df_keys) != len(df_counts)):
                return None
            enc_ = GuardEncoder.from_df(dict(zip(df_keys, df_counts)), meta["n_texts"], dim)
            return enc_, ProtoStore.from_state(meta, {"A": A, "P": P})
        except Exception:
            return None

    got = _load()
    if got is not None:
        return got[0], got[1], "loaded"
    # ONE BUILDER AT A TIME. MEASURED: tests/test_guide_examples.py runs 8 fresh processes at once, and with a cold
    # cache all 8 built the rows together on 2 cores -- 26.6 s, over the 15 s budget (a skip), where one build is
    # ~2.5 s. So the first process takes a lock file and builds; the others wait for its file (polling) instead of
    # competing for the same CPU. A lock older than GUARD_CACHE_WAIT_S is a builder that died: ignored. A wait that
    # runs out, or a temp dir that refuses the lock, just builds -- the lock only saves work, it never gates a result.
    import time
    lock, own = path + ".lock", False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        own = True
    except FileExistsError:
        try:
            fresh = time.time() - os.path.getmtime(lock) < GUARD_CACHE_WAIT_S
        except OSError:
            fresh = False                             # the builder just finished and removed it
        deadline = time.time() + GUARD_CACHE_WAIT_S
        while fresh and time.time() < deadline and os.path.exists(lock) and not os.path.exists(path):
            time.sleep(0.05)
        got = _load()
        if got is not None:
            return got[0], got[1], "loaded"
    except OSError:
        pass                                          # no lock possible here: build without one
    try:
        enc, st = build_typed_store(examples, epochs=epochs, dim=dim)
        try:
            meta, arrays = st.state(dtype="float64")
            meta["cache_key"], meta["n_texts"] = key, enc.n_texts
            keys = list(enc.df)
            tmp = "%s.%d.tmp" % (path, os.getpid())
            with open(tmp, "wb") as f:
                np.savez(f, A=arrays["A"], P=arrays["P"],
                         meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8),
                         df_keys=np.frombuffer(json.dumps(keys).encode("utf-8"), dtype=np.uint8),
                         df_counts=np.asarray([enc.df[g] for g in keys], dtype=np.int64))
            os.replace(tmp, path)
        except OSError:
            pass                                      # a read-only temp dir costs the next process a build, nothing else
    finally:
        if own:
            try:
                os.remove(lock)
            except OSError:
                pass
    return enc, st, "built"


def type_scores(store, x):
    """{type: best row cosine} for one encoded question (-1 for a type with no rows)."""
    s = store.scores(x)
    out = {t: -1.0 for t in TYPED_OPTIONS}
    for j, lab in enumerate(store.labels):
        t = store.key_of.get(lab)
        if t in out and s[j] > out[t]:
            out[t] = float(s[j])
    return out


def typed_value(scores, margin=None, floor=None):
    """The decision rule on {type: score}: the non-ordinary type with the largest lead over ordinary beyond its
    margin, else 'ordinary'. A credential or live_value decision also needs its own row cosine >= floor; when the
    only types that led are UNDER the floor -- the question leans that way but resembles little the guard has seen --
    the answer is 'unclear' (ask), not a guess either way. unclear has no floor: a bare question resembles little by
    definition. Returns (value, {type: lead over ordinary}).
    MEASURED on the train half (--cv, safety 0.03, floor 0.18; caught credential / live / unclear of 79 / 286 / 29):
    under-the-floor -> ordinary 62 / 245 / 27, -> unclear 63 / 250 / 27, both 0 train and 0 leave-one-fold-out false
    positives, 0 on the catalog tuning half x3 and the tests/guides facts. Adopted AFTER the second held-out run, whose
    B/C readings fell 32 -> 24/40 with the floor (short slang like "eth rn" has few features, so low cosines) -- so B/C
    are not clean for this one change; the train half above is the evidence it stands on."""
    margin = TYPED_MARGIN if margin is None else margin
    floor = GUARD_FLOOR if floor is None else floor
    lead = {t: round(scores[t] - scores["ordinary"], 4) for t in ("credential", "live_value", "unclear")}
    best, over, weak = "ordinary", 0.0, False
    for t in ("credential", "live_value", "unclear"):             # fixed order: the one stated tie rule
        o = lead[t] - margin[t]
        if o > 0 and o > over:
            if t == "unclear" or scores[t] >= floor:
                best, over = t, o
            else:
                weak = True                                        # led, but on too little resemblance
    if best == "ordinary" and weak:
        best = "unclear"
    return best, lead


class SemanticGuard:
    """The TYPED learning guard (E4.2): is a question a request for a CREDENTIAL, a LIVE VALUE, too bare to tell
    (UNCLEAR), or ORDINARY? The base rows are built once per process from the shipped examples; each mind's guard
    adds the examples IT learned as exemplar rows on top. Public API (unchanged since sweep 180, so the mixins
    need no change): intent(q), learn(q, kind), state(), restore(st); new: decide(q)."""

    _BASE = None                                   # (encoder, trained ProtoStore) -- per process, read-only

    def __init__(self):
        from holographic.agents_and_reasoning.holographic_protostore import DoorCalibrator, TeacherNoise
        self.learned = {"credential": [], "reading": [], "normal": [], "unclear": []}
        self._store = None                         # base rows + learned exemplar rows (built lazily)
        # this door's own label stream: every learn() is a labelled verdict on the typed decision it replaces
        self.calibration = DoorCalibrator("learnguard", min_count=8)
        # re-labels of the same question estimate how often the teacher (the pattern layer, a person, a model)
        # flips; until 20 exist eps_hat is 0 and the update is exactly the plain rule
        self.noise = TeacherNoise(m=3, min_pairs=20)

    # how the base rows arrived in this process: None (not yet), 'loaded' (the disk cache), 'built', or
    # 'built (cache off)' -- a structural receipt tests read instead of timing the build (tests/test_perf_budget.py)
    _BASE_HOW = None

    @classmethod
    def _base(cls):
        """(encoder, base ProtoStore): built ONCE per process and shared read-only by every mind's guard (each mind
        deep-copies it before adding its own learned rows -- 2.3 ms). Through cached_typed_store, so a process whose
        temp dir already holds the rows for this exact code and example list loads them instead of rebuilding."""
        if cls._BASE is None:
            enc, st, how = cached_typed_store(shipped_examples())
            cls._BASE, cls._BASE_HOW = (enc, st), how
        return cls._BASE

    def _rows(self):
        """The live store: a copy of the base rows, then every learned example replayed IN A FIXED ORDER --
        credential, reading, unclear, then normal LAST, so a correction ('this is ordinary') is applied after the
        refusals it corrects."""
        if self._store is None:
            import copy
            enc, base = self._base()
            self._store = copy.deepcopy(base)
            for kind in ("credential", "reading", "unclear", "normal"):
                for q in self.learned[kind]:
                    self._add_exemplar(enc, q, _TYPE_OF_KIND[kind], eps=0.0)
        return self._store

    def _add_exemplar(self, enc, q, typ, eps):
        x = enc(q)
        lab = "learned:%s:%s" % (typ, _hashlib.sha256(q.encode("utf-8")).hexdigest()[:12])
        self._store.add_option(lab, x[None, :], key=typ)
        self._store.update(x, lab, eps=eps)        # the contrastive step: rows of other types move away from q

    def decide(self, question):
        """THE TYPED DECISION: {value, ranked: [(type, score)], lead: {type: score - ordinary}, confidence,
        p_correct}. value is one of TYPED_OPTIONS; 'unclear' means the caller should ask what was meant."""
        enc, _ = self._base()
        sc = type_scores(self._rows(), enc(question))
        value, lead = typed_value(sc)
        ranked = sorted(sc.items(), key=lambda kv: (-kv[1], TYPED_OPTIONS.index(kv[0])))
        conf = round(ranked[0][1] - ranked[1][1], 4)
        return {"value": value, "ranked": [(t, round(s, 4)) for t, s in ranked], "lead": lead,
                "confidence": conf, "p_correct": self.calibration.p_correct(conf)}

    def intent(self, question):
        """{credential, reading, unclear}: how far each type leads ORDINARY for this question (> 0 means closer
        to that type than to anything ordinary; the verdict needs more -- TYPED_MARGIN). The sweep-180 keys."""
        d = self.decide(question)["lead"]
        return {"credential": d["credential"], "reading": d["live_value"], "unclear": d["unclear"]}

    def learn(self, question, kind):
        """Add one labelled example. kind: credential / reading (= live_value) / unclear / normal (= ordinary).
        A 'normal' example is THE correction for a false positive; learning a question under one kind removes it
        from the others. Returns True when something was added."""
        if kind not in _TYPE_OF_KIND:
            raise ValueError("kind must be one of %s" % sorted(_TYPE_OF_KIND))
        typ = _TYPE_OF_KIND[kind]
        k = _KIND_OF_TYPE[typ]
        # redact first (the learning-loop audit, 2026-09-27): the guard learns the question it just REFUSED, and
        # that question can carry the very secret it refused -- keep the wording, never the value
        q = normalise_question(redact(question))
        if not q:
            return False
        before = [kk for kk in self.learned if q in self.learned[kk]]
        # a labelled verdict for this door's calibrator: was the typed decision already right?
        d = self.decide(q)
        self.calibration.observe(d["confidence"], d["value"] == typ)
        for kk in before:                          # a re-label: the teacher agreeing with itself, or flipping
            self.noise.observe(kk, k)
        if before == [k]:
            return False
        moved = [kk for kk in before if kk != k]
        for kk in moved:
            self.learned[kk].remove(q)
        self.learned[k].append(q)
        dropped = len(self.learned[k]) > MAX_LEARNED
        del self.learned[k][:-MAX_LEARNED]         # bounded: the oldest drop first
        if moved or dropped or self._store is None:
            self._store = None                     # a removal cannot be un-applied to the rows: rebuild lazily
        else:
            enc, _ = self._base()
            self._add_exemplar(enc, q, typ, eps=self.noise.eps_hat())
        return True

    def state(self):
        """JSON-able: the learned TEXTS per kind (the sweep-180 format plus 'unclear'), and this door's
        calibration and noise pairs. Rows are never stored -- restore() rebuilds them from the texts."""
        st = {k: list(v) for k, v in self.learned.items()}
        st["calibration"] = self.calibration.state()
        st["noise"] = self.noise.state()
        return st

    def restore(self, state):
        """Read either format: sweep 180/181 ({credential, reading, normal}) or E4.2 (plus unclear, calibration,
        noise). Unknown keys are ignored."""
        from holographic.agents_and_reasoning.holographic_protostore import DoorCalibrator, TeacherNoise
        state = state or {}
        for k in self.learned:
            v = state.get(k, [])
            self.learned[k] = [str(x) for x in (v if isinstance(v, (list, tuple)) else [])][-MAX_LEARNED:]
        if isinstance(state.get("calibration"), dict):
            self.calibration = DoorCalibrator.from_state(state["calibration"])
        if isinstance(state.get("noise"), dict):
            self.noise = TeacherNoise.from_state(state["noise"])
        self._store = None


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def _credential_shaped(answer):
    """A bare value: one whitespace-free token of 6+ characters that is not a path, a URL or an email address.
    An EMAIL is an identifier, not a secret (found by tests/test_guide_examples.py in sweep 181: the semantic
    layer refused "my email" -> an address at intent 0.0135, a hair over the margin; the owner's list of what
    must never be learned is keys, seed phrases and passwords). Sharing PII is contribute()'s screen, not this."""
    a = str(answer).strip().strip("\"'`")
    return (bool(a) and not any(c.isspace() for c in a) and len(a) >= 6 and "/" not in a and "\\" not in a
            and not _EMAIL.match(a))


# The pattern layer's _MECHANISM is deliberately broad ("how is ..." / "how does ..." anything), because the pattern
# layer has no other way to tell "how is the pool price computed" from a reading. The TYPED decision already learned
# mechanism talk as ordinary ("how is tvl calculated", "how does a weather forecast work"), so its answer gate only
# needs to drop TRUE mechanism questions -- the broad rule was throwing away "how is the traffic on the way to the
# beach" and "how does the traffic look on my way to work right now". MEASURED on the train half (5-fold CV, 0 FP):
# live 245/286 -> 251/286, no new false positive.
_MECHANISM_TYPED = re.compile(
    r"^\s*(why\b|explain\b|how (?:can|could|should|would|to)\b|"
    r"how (?:is|are|does|do|did)\b.*\b(?:computed|calculated|set|determined|made|measured|priced|derived|decided|"
    r"chosen|picked|updated)\b|how (?:does|do)\b.*\bwork\s*[?.!]*$|"
    r"what (?:does|do) .+ mean\b|what is the (?:formula|definition|meaning)\b)", re.I)


# A NUMBER-LED value: a digit at the start of a token (or after a currency sign / bracket / sign) -- "$0.92", "72F",
# "3.2 SOL", "1 USD = 0.92 EUR". A digit glued to a name is not a reading: "AbsRec@1", "orient2d", "p29", "sha256".
_NUMBER_LED = re.compile(r"(?:^|[\s$€£¥(~≈+]|(?<![A-Za-z0-9])-)\d")   # a '-' is a SIGN only when not glued
                                                                        # to a word ("payload-1-0" is a name)
_WEATHER_WORD = re.compile(r"\b(sunny|cloudy|overcast|rain(?:y|ing)?|snow(?:y|ing)?|clear|storm(?:y)?|fog(?:gy)?|windy|"
                           r"drizzle)\b", re.I)


def _reading_shaped(question, answer):
    """A short measured answer -- a number-led value or a weather word -- to a question with no time anchor that is
    not about a MECHANISM (the narrow rule above) or a route: what a live_value decision needs before it refuses.
    (The pattern layer's _MEASUREMENT takes ANY digit; found on the catalog tuning half after the held-out run: a card
    name "Paired accuracy + AbsRec@1 ..." then counted as a reading and pinned the live margin at 0.11.)"""
    q = str(question or "")
    if q.startswith(ROUTE_PREFIXES) or _ANCHOR.search(q) or _MECHANISM_TYPED.match(q.strip()):
        return False
    a = str(answer or "")
    return len(a.split()) <= MAX_READING_WORDS and bool(_NUMBER_LED.search(a) or _WEATHER_WORD.search(a))


def _passphrase_shaped(answer):
    """An all-lower-case run of 12+ letters ("correcthorsebatterystaple"): the passphrase the pattern layer's
    password rule was widened for. Method names have underscores and card names have spaces, so neither matches."""
    return bool(re.fullmatch(r"[a-z]{12,}", str(answer or "").strip().strip("\"'`")))


def _pin_shaped(answer):
    """A PIN: 4 to 8 digits and nothing else. Too short for _credential_shaped (6+), and a number like "8080" is an
    ordinary answer to most questions -- so it only counts where the TYPED decision already says credential
    ("the pin for my visa card" -> "4821"); before E4.2 that pair was learned."""
    return bool(re.fullmatch(r"\d{4,8}", str(answer or "").strip()))


_VALUEISH = re.compile(r"^[\s$€£¥~≈+-]*[\d][\d.,:]*\s*[%a-zA-Z/°]{0,8}\.?$")


def _secret_shaped(answer):
    """A bare token that could be a secret: credential-shaped AND a digit, a non-word symbol or an internal capital.
    What a question typed CREDENTIAL needs from its answer: "q8Zr4Lw2Pn", "hunter22" and "correcthorseFAKE" are; a
    method name ("place_work", "radiance_transfer") or a card name ("Voxelization") is not. FOUND AFTER THE HELD-OUT
    RUN on the catalog tuning half, with each alias answered by its card's name and method (what decision_outcome
    teaches): with any bare token counting, engine phrases like "one answer for where to run" -> "place_work" pinned
    the credential margin at 0.21 (train-half credential 38/79)."""
    a = str(answer or "").strip().strip("\"'`")
    return _credential_shaped(a) and bool(re.search(r"\d", a) or re.search(r"[^\w]", a) or
                                          (re.search(r"(?<=.)[A-Z]", a) and re.search(r"[a-z]", a)))


# A MEASUREMENT token (number-led, see _NUMBER_LED): an amount ("$0.41"), a decimal ("0.97", "2.1 SOL"), a percent,
# a number with a short unit ("72F", "31 gwei"), a clock time ("3:45"), or 4+ digits ("4821"). A lone small integer
# after a word is a label, not a reading: tests/test_learning_dedupe.py teaches "question 0" -> "answer 0" and the
# first rule (any number-led value) refused it as unclear -- found by that test after the held-out runs.
_VALUE_TOKEN = re.compile(r"(?:^|[\s(~≈+]|(?<![A-Za-z0-9])-)(?:[$€£¥]\s?\d|\d[\d,]*\.\d|\d+(?:[.,]\d+)?\s?%|"
                          r"\d{1,2}:\d{2}|\d{4,}|\d+(?:[.,]\d+)?\s?[A-Za-z]{1,5}\b)")


def _strong_secret(answer):
    """A token that looks RANDOM, not like a name: credential-shaped with a digit AND both letter cases, or a digit
    and a password symbol. "q8Zr4Lw2Pn", "Hunter2-FAKE-9c1d" and "9f3Ab!x7" are; "orient2d", "hunter22" and
    "holographic_spatial.knn" are not. Only the UNCLEAR path needs this much from the answer (the question says
    nothing); a question typed credential takes any secret-shaped token."""
    a = str(answer or "").strip().strip("\"'`")
    if not _credential_shaped(a) or not re.search(r"\d", a):
        return False
    return bool((re.search(r"[a-z]", a) and re.search(r"[A-Z]", a)) or re.search(r"[!#$%^&*+=?~]", a))


def _bare_value(answer):
    """What an UNCLEAR question must be answered with before it is refused: at most 3 words holding a MEASUREMENT
    token (_VALUE_TOKEN: "$0.41", "2.1 SOL", "0.97", "4821"), or a strongly secret-looking token
    ("q8Zr4Lw2Pn"). "ada?" -> "$0.41" teaches nothing reusable and may be a stale reading, so it clarifies instead.
    FOUND AFTER THE HELD-OUT RUN (kept loud): the first rule ("3 words with any digit, or any bare token") refused 5 of
    the 144 facts the repo's tests and guides teach ("q1" -> "a1", "a" -> "1", "my email" -> "owner0@example.com")
    and, on the catalog tuning half, bare engine terms answered with identifiers ("orient2d" -> "orient2d",
    "euclidean" -> "holographic_spatial.knn")."""
    a = str(answer or "").strip()
    if not a or _EMAIL.match(a):
        return False
    if len(a.split()) <= 3 and _VALUE_TOKEN.search(a):
        return True
    return _strong_secret(a)


# ---- the one verdict every door asks for ------------------------------------------------------------------

def learning_verdict(question, answer="", allow_volatile=False, guard=None):
    """THE VERDICT: {ok, kind: None|'sensitive'|'volatile', reason, layer}. Sensitive has no override. (Named
    learning_verdict, not `check`: that name is taken by two other modules -- the duplication audit.)
    PATTERN layer FIRST (shapes and cue words; unchanged by E4.2 -- strong shapes always refuse). Only where it is
    silent, and only when `guard` (a SemanticGuard) is given, the TYPED decision (credential / live_value /
    unclear / ordinary) plus the answer's shape. A typed refusal carries typed=<value>; an unclear one also
    clarify=True -- the caller should ask what was meant. A pattern refusal with a question-side cue teaches the
    typed guard its type, so the next paraphrase is caught by meaning."""
    r = sensitive_pair_reason(question, answer)
    if r:
        if guard is not None and _SECRET_CUE.search(str(question)):
            guard.learn(question, "credential")
        return {"ok": False, "kind": "sensitive", "reason": r, "layer": "pattern"}
    if not allow_volatile:
        r = volatile_reason(question, answer)
        if r:
            if guard is not None:
                guard.learn(question, "reading")
            return {"ok": False, "kind": "volatile", "reason": r, "layer": "pattern"}
    if guard is None:
        return {"ok": True, "kind": None, "reason": None, "layer": None}
    if not answer_shapes(question, answer, allow_volatile):
        return {"ok": True, "kind": None, "reason": None, "layer": None}   # no answer shape: nothing to decide
    d = guard.decide(question)
    return typed_verdict(question, answer, d["value"], d["lead"], allow_volatile, p_correct=d["p_correct"]) or \
        {"ok": True, "kind": None, "reason": None, "layer": None}


def answer_shapes(question, answer, allow_volatile=False):
    """Which typed refusals this ANSWER could support: {'credential', 'live_value', 'unclear'} subset. Empty means
    the typed decision cannot refuse anything here, so it is not even computed."""
    out = set()
    if _secret_shaped(answer) or _pin_shaped(answer) or _passphrase_shaped(answer):
        out.add("credential")
    if not allow_volatile and _reading_shaped(question, answer):
        out.add("live_value")
    if _bare_value(answer):
        out.add("unclear")
    return out


def typed_verdict(question, answer, value, lead, allow_volatile=False, p_correct=None):
    """The refusal a typed decision supports for this (question, answer), or None. Split out of learning_verdict so
    tools/bench_learnguard.py --cv judges held-in folds with EXACTLY the shipped rule."""
    shapes = answer_shapes(question, answer, allow_volatile)
    base = {"ok": False, "layer": "semantic", "typed": value, "intent": (lead or {}).get(value),
            "p_correct": p_correct}
    if value == "credential" and "credential" in shapes:
        return dict(base, kind="sensitive",
                    reason="sensitive (semantic): the question reads as a request for a credential and the answer is "
                           "a bare value -- never learned; if this is wrong, learn_guard_example(question, 'normal')")
    if value == "live_value" and "live_value" in shapes:
        return dict(base, kind="volatile",
                    reason="volatile (semantic): the question reads as a request for a live reading -- learn the tool "
                           "that fetches it, or a dated snapshot, or pass allow_volatile=True; if this is wrong, "
                           "learn_guard_example(question, 'normal')")
    if value == "unclear" and "unclear" in shapes:
        # a bare VALUE (a number, an amount) is the volatile side and allow_volatile lets it through; a bare TOKEN
        # that is not a number could be a secret and is refused as one
        valueish = bool(_VALUEISH.match(str(answer).strip())) or "credential" not in shapes
        if valueish and allow_volatile:
            return None
        kind = "volatile" if valueish else "sensitive"
        return dict(base, kind=kind, clarify=True,
                    reason="%s (semantic): unclear -- the question is too bare to tell what it asks (a bare ticker or "
                           "'the code?'); ask what was meant instead of storing a bare value; if this is wrong, "
                           "learn_guard_example(question, 'normal')" % kind)
    return None


# ---- redaction: for text that must TRAVEL (records, previews, ingested documents) but must not carry a secret

def redact(text):
    """The text with every sensitive span replaced by [redacted]. Strong shapes and BIP-39 runs always; the
    value of a `password: ...` assignment; weak shapes only when a credential word is present."""
    s = str(text)
    for _, rx in _STRONG:
        s = rx.sub("[redacted]", s)
    span = _seed_run(s)
    while span:
        s = s[:span[0]] + "[redacted]" + s[span[1]:]
        span = _seed_run(s)
    # The VALUE is the last group of every assignment / disclosure pattern (group 2 of the older ones; the
    # value-first and login-with wordings have it as group 1). Replace right-to-left over de-overlapped spans, so
    # two patterns catching the same value never splice into each other's indices.
    spans = sorted({(m.start(m.lastindex), m.end(m.lastindex)) for m in _assignment_hits(s)}, reverse=True)
    last_start = None
    for a, b in spans:
        if last_start is not None and b > last_start:
            continue
        s = s[:a] + "[redacted]" + s[b:]
        last_start = a
    if _SECRET_CUE.search(s):
        for _, rx in _WEAK:
            s = rx.sub("[redacted]", s)
    return s


def redact_args(obj):
    """A copy of a tool call's arguments safe to record: any KEY that names a credential has its value replaced,
    and every string value is run through redact(). Structure is kept so the record still reads."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _SECRET_CUE.search(str(k)) or re.search(r"(?i)(^|_)(key|token|secret|pass|pwd|seed)($|_)", str(k)):
                out[k] = "[redacted]"
            else:
                out[k] = redact_args(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_args(v) for v in obj)
    if isinstance(obj, str):
        return redact(obj)
    return obj


def _selftest():
    check = learning_verdict
    fake_key = "sk-live-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"
    seed = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    assert check("what is my exchange api key", fake_key)["kind"] == "sensitive"
    assert check("my wallet seed phrase", seed)["kind"] == "sensitive"
    assert check("the admin password", "hunter2-Moose!")["kind"] == "sensitive"
    assert check("how do I reset my password", "open settings, then security, then reset")["ok"]
    assert check("where is my solana keypair", "~/.config/solana/id.json")["ok"]
    assert check("what is the current price of solana", "$142.10")["kind"] == "volatile"
    assert check("what is the weather right now", "72F and sunny")["kind"] == "volatile"
    assert check("sol price as of 2026-09-22 15:00", "$142.10")["ok"]                   # a dated snapshot
    assert check("what is the current price of solana", "$142.10", allow_volatile=True)["ok"]
    assert check("what is solana", "a layer-1 blockchain")["ok"]                          # no measurement
    assert check("what port does the leCore service use", "8080")["ok"]                  # no cue
    assert check("toolreflex: price of sol now", '{"service": "px", "endpoint": "sol"}')["ok"]
    assert "[redacted]" in redact("key=" + fake_key) and fake_key not in redact("use " + fake_key)
    assert redact_args({"api_key": fake_key, "symbol": "SOL"}) == {"api_key": "[redacted]", "symbol": "SOL"}
    tx = "5VERv8NMvzbJMEkV8xnrLkEaWRtSz9CosKDYVCJjBRAHYjiQnbzVbZ1hASUnUWdN9Kv2LHhqpPmGbsW3PPkpk2VL"
    assert check("what was the signature of that transaction", tx)["ok"]                  # public, same shape
    assert check("what is the wifi password", "correcthorsebatterystaple")["kind"] == "sensitive"   # probed miss
    assert check("which password manager do you use", "bitwarden")["ok"]
    g = SemanticGuard()
    assert check("spec 9 of the flux rotor", "rated 9.4 kilowatts", guard=g)["ok"]   # a real false positive
    assert check("what is lever 7", "the displacement trace, sweep 104", guard=g)["ok"]
    assert check("secret 0 of session 1", "payload-1-0")["ok"]      # test_mcp_server release audit
    # the two calibration false positives, pinned
    assert check("what did the release sweep verify", "PASS: rack_roundtrip; pass: lever_heal ok")["ok"]
    assert check("what is the install pipeline now", " ".join(["the"] * 40) + " 3 stages")["ok"]
    assert check("notes", " ".join(["step"] * 40))["ok"]            # one repeated BIP-39 word is not a seed
    assert check("how is the pool price computed", "price = y / x for a constant-product pool; " * 3 + "fee 0.3%")["ok"]
    assert check("what time is it right now", "3:45 pm")["kind"] == "volatile"
    assert check("how much is bitcoin worth right now", "$64000")["kind"] == "volatile"   # how MUCH: a value
    # E4.2 -- the typed layer: four options, PIN talk is ordinary, a bare referent is unclear and clarifies
    assert g.decide("the pin for my visa card")["value"] == "credential"
    assert g.decide("I forgot my passcode, what do I do")["value"] == "ordinary"
    assert g.decide("the code?")["value"] == "unclear"
    v = check("the code?", "q8Zr4Lw2Pn", guard=g)
    assert v["kind"] == "sensitive" and v["clarify"] is True and v["typed"] == "unclear"
    assert check("how many yen for a dollar", "147.2", guard=g)["kind"] == "volatile"
    assert check("how many yen for a dollar", "147.2", allow_volatile=True, guard=g)["ok"]
    # a correction wins, and the learned state round-trips as text (the sweep-180 format plus 'unclear')
    assert g.learn("how many yen for a dollar", "normal") and check("how many yen for a dollar", "147.2", guard=g)["ok"]
    g2 = SemanticGuard()
    g2.restore(g.state())
    assert check("how many yen for a dollar", "147.2", guard=g2)["ok"]
    g2.restore({"credential": [], "reading": ["how many yen for a dollar"], "normal": []})     # a sweep-181 state
    assert check("how many yen for a dollar", "147.2", guard=g2)["kind"] == "volatile"
    print("holographic_learnguard selftest OK -- secrets refused, readings refused, snapshots and routes kept, "
          "typed guard decides and learns")


if __name__ == "__main__":
    _selftest()
