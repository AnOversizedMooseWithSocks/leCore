"""holographic_rolecall.py -- method calls, options and trajectories as ROLE-FILLER hypervectors; compose a call
from known pieces with a chimera check; read a slot's DIRECTION from the question's words (CLM backlog, phase E).

WHY THIS EXISTS (docs/BACKLOG_contrastive.md, E3.1 / E3.2 / E5.1)
------------------------------------------------------------------
Ranking stays in the sparse space (hypervectors cost 0.9 top-1 points against exact sparse ranking, measured by
the CLM panel). The hypervector space earns its keep where sparse vectors cannot go: COMPOSING and DECOMPOSING
structure. This module is that half, as one encoding spec with its decoders and its honesty checks:

  * A METHOD CALL is a SUM of role-filler pairs:          VERB(x)v + sum_k SLOT_k(x)x_k
      HRR bind (circular convolution) at D=2048 with UNITARY role atoms named by slot, so unbinding a role is
      EXACT (the involution is the true inverse of a unitary atom). The panel measured per-role unbind + cleanup
      at 500/500 (D=1024 and 2048), 0.1 ms, robust to Gaussian noise equal to each term's norm
      (docs/research/evidence/clm_panel_20260926/w2-hd/exp_a_resonator.py).
  * A FREE-TEXT candidate sits under a TEXT role:          TEXT(x)ngrams(text)
  * A TYPED OPTION is                                      OPTION(x)name + TEXT(x)ngrams(examples)
  * A TRAJECTORY is a permutation-ordered bundle:          sum_t rho^t(step_t)
  * A PRODUCT form (one bound product, for a resonator) needs PERMUTATION roles: a MAP product is commutative, so
      FROM*x*TO*y == FROM*y*TO*x and bound role atoms cannot carry order inside one product (panel: order right
      50% at D=2048 / 27% at D=1024 -- no better than no roles); rho^1(x) * rho^2(y) decoded 30/30 and 29/30.

WHAT CANNOT BE READ FROM THE ENCODING, and where it is read instead: DIRECTION. A candidate call FROM(x)USD +
TO(x)EUR is perfectly distinct from its swap (panel: swapped-call cosine 0.024 +/- 0.026, noise p99 0.056), but
WHICH currency the person has is a fact about the QUESTION's words ("how many pesos for one dollar" names the TO
currency first). ContextRoles learns, per slot NAME, a prototype of the CONTEXT a filler appears in (the word before
it, two before, the word after -- each bound to its own unitary context role) from `from_question` verdicts, and
reads the direction of a new question from its context words rather than the order the currencies appear. The
panel's small, partly rule-labelled set: 26/26 held out vs positional 21/26. The hand-labelled set
(tests/data/exchange_direction.json) and tools/bench_rolecall.py are the real measurement.

THE CHIMERA CHECK. A readout that BLENDS two calls decodes, role by role, into a mix of both 74-76% of the time
(panel, equal 2-call blends): the sum form keeps role-filler PAIRS, not which pairs belong together, so a per-role
argmax can build a call nobody ever made. compose() therefore does not stop at the argmax: it RECOMPOSES the decoded
call, compares it with the state, EXPLAINS IT AWAY (subtracts it) and re-reads every role in the residual. Nothing
left -> 'clean'; a rival left but beaten by a margin in noise units -> 'dominant'; a rival as strong as the pick ->
'ambiguous' (the chimera risk: the grouping is not in the state and must not be guessed).

SCOPE, stated honestly: compose() proposes NEW COMBINATIONS of KNOWN verbs and values -- it cannot invent a verb or
a value it has no atom for. A proposal is a CANDIDATE with a confidence, never an action: nothing here calls a tool.

MEASURED (tools/bench_rolecall.py -> docs/research/evidence/bench_rolecall.json; machine load ~9 on 2 cores)
  * DIRECTION, hand-labelled exchange set (179 items: 100 CLINC150 + 79 hand-written chat / LLM-agent wordings;
    tests/data/exchange_direction.json), held-out hash split: context roles 49/51 vs positional 35/51 (discordant
    15 vs 1; CLINC150 31/31 vs 19/31). 5-fold CV over all 179: 172 vs 111. Learning curve: 4 taught 33/51, 16
    taught 46/51, 32 taught 49/51 (flat after). Ablations all 48-49/51 (no mask, L1+R1, +R2, infonce rule).
  * COMPOSE, synthetic (panel replication, 20 verbs x 50 values, 500 trials): clean and noise-as-long-as-the-state
    500/500 served right at D=1024 and 2048; EQUAL 2-call blends: per-role argmax builds chimeras 71.2% (D=2048) /
    74.0% (D=1024) -- the panel's 74-76% -- compose() flags 100% ambiguous and serves 0 chimeras; DOMINANT 0.7/0.3
    blends: 100% served right, 0 flagged.
  * COMPOSE, real wording (5-fold CV, reader never saw the question): 172/179 calls right; on the 102 calls NEVER
    SEEN WHOLE in the training folds 96/102 right, ECE 0.026 (cross-fitted isotonic p) / 0.062 (cold-start
    p_model); the 23 with a (from, to) pair never seen: 23/23. A compose is 2-3 ms at D=2048 (50 values, 20 verbs)
    and a whole question (read + soft state + compose) ~9 ms, both under load ~9 on 2 cores.
  * Held-out SYMMETRIC sentences ('rate between usd and cad'): 14/14 composed as 'ambiguous' once 27 'either'
    verdicts are taught (0/14 without them), 0 of 51 directional test items called directionless.

KEPT NEGATIVES (measured, kept loud)
  * The 7 wrong real-wording calls are ALL the amount-in-the-TO-currency construction ('buy 200 euros WITH
    dollars', 'need to pay 3000 euros; how many dollars'): the <num> before a currency is the strongest FROM cue,
    and 3 context words cannot see 'buy ... with'. They are wrong CONFIDENTLY (p 0.96-0.997): the calibrator is
    right on average (ECE 0.026) but cannot rank these below the right ones -- more taught examples of the
    construction, not a better threshold, is the fix.
  * Taught only directional verdicts, the reader is CONFIDENTLY directional on symmetric sentences (0/14 flagged,
    reader p 0.98): 'no direction' must be taught ({'either': [...]}) -- it cannot be inferred from a window.
  * The isotonic calibrator clamps below its lowest seen score, so an 'ambiguous' candidate read p 0.83; the mind
    faculty caps it by p_model (0.30 on the symmetric probe).
  * A 1-sd dead zone in p_model's coefficient share cost a clean 3-role call over 50 values its confidence (0.72);
    2 sd is used. The residual's own pick (its estimation error) read as a 12-sd 'rival' until excluded.

numpy + stdlib only, deterministic (blake2b/sha256-derived atoms, seeded generators, stable sorts; ties keep
insertion / positional order -- the one stated tie rule).
"""
import hashlib
import itertools
import json
import math
import re

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import Vocabulary, bind, cosine, derived_atom, permute, unbind
from holographic.sampling_and_signal.holographic_fft import irfft as _irfft, rfft as _rfft   # bind's own FFT backend

NONE_SLOT = "either"  # the slot name a no-direction verdict teaches ({'either': [words, words]})
DIM = 2048          # the panel's measured operating point (500/500 decode at 1024 and 2048; 2048 has the margin)
SEED = 0

# ============================================================================================ tokens and numbers
# ONE tokenizer for word spans, shared with tests/data/exchange_direction.json (its split_rule field states it):
# lowercase; a number with thousands commas / decimals is ONE token ("20,000", "1,250.00"); currency signs are
# tokens of their own ("$30" -> "$", "30"), because '$' is sometimes the ONLY word that names the FROM currency
# ("in canadian dollars, what is $30").
_TOK = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+|[$€£¥]")

# single number words people type instead of digits (the ones measured in CLINC150's exchange phrasings)
_NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
             "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100"}
_SCALE = {"k": 1000, "thousand": 1000, "million": 1000000, "m": 1000000, "billion": 1000000000}


def question_tokens(text):
    """The word tokens spans are defined over: lowercase, numbers kept whole, currency signs as tokens."""
    return _TOK.findall(str(text).lower().replace("’", "'"))


def find_span(tokens, words, taken=()):
    """First occurrence of `words` (a string, tokenized the same way) in `tokens` that does not overlap a span in
    `taken` -> (start, end) token indices, or None. Earliest wins (the one tie rule)."""
    w = question_tokens(words)
    if not w:
        return None
    for i in range(len(tokens) - len(w) + 1):
        if tokens[i:i + len(w)] == w and not any(i < b and a < i + len(w) for a, b in taken):
            return (i, i + len(w))
    return None


def _norm_number(tok, nxt=None):
    """'1,250.00' -> '1250.00'; '10' + 'k' -> '10000'; '2.5' + 'million' -> '2500000'. Returns a string."""
    s = tok.replace(",", "")
    if nxt in _SCALE:
        v = float(s) * _SCALE[nxt]
        return str(int(v)) if v == int(v) else repr(v)
    return s


def amounts_in(text):
    """Numbers a question gives, normalised, in order: digits (commas and decimals kept whole, a trailing k /
    thousand / million scale applied) and single number words ('five dollars'). A bare 'a' / 'an' is NOT a
    number here -- 'a dollar' is read as 1 only by the caller that knows it names a currency."""
    toks = question_tokens(text)
    out = []
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t[0].isdigit():
            out.append(_norm_number(t, nxt))
        elif t in _NUMWORDS:
            out.append(_NUMWORDS[t])
    return out


def _phi(x):
    """Standard normal CDF (math.erf, no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def call_string(verb, args):
    """Canonical text of a call: 'fx(amount=250, from=EUR, to=JPY)' -- sorted slots, so one call has one
    spelling (it is what a decision id and an outcome are compared on)."""
    inner = ", ".join("%s=%s" % (k, args[k]) for k in sorted(args))
    return "%s(%s)" % (verb, inner)


# ============================================================================================ the role codec
class RoleCodec:
    """The encoding spec (E3.1) and the sum-form decoder with its chimera check (E3.2).

    Role atoms are UNITARY and derived from (seed, 'role:<name>'): a role named 'from' is the same vector in every
    process, and unbinding it is exact. Fillers are ordinary Gaussian atoms derived from (seed, '<kind>:<name>')
    through holographic_ai.Vocabulary(derived=True), so a verb, a value and a number never collide by name."""

    VERB, TEXT, OPTION = "VERB", "TEXT", "OPTION"

    def __init__(self, dim=DIM, seed=SEED):
        self.dim, self.seed = int(dim), int(seed)
        self._roles = Vocabulary(self.dim, seed=self.seed, unitary=True, derived=True)
        self._fill = Vocabulary(self.dim, seed=self.seed, derived=True)
        self._text_enc = None
        self._null_cache = {}
        self._mats = {}
        self._specs = {}

    # ---------------------------------------------------------------- atoms
    def role(self, name):
        """The unitary role atom for a slot name (or VERB / TEXT / OPTION)."""
        return self._roles.get("role:" + str(name))

    def atom(self, value, kind="val"):
        """The filler atom for a value; kind namespaces it ('val', 'verb', 'num', 'opt')."""
        return self._fill.get("%s:%s" % (kind, value))

    def _filler(self, role, value):
        """Which namespace a role's filler lives in: the VERB role holds verbs, 'amount' holds numbers."""
        if role == self.VERB:
            return self.atom(value, "verb")
        if role == self.OPTION:
            return self.atom(value, "opt")
        return self.atom(value, "val")

    def text_vec(self, text):
        """A unit hashed character-n-gram vector for free text (the encoder the typed decisions and RecordCodec use,
        holographic_systemone.hashed_ngram_encode)."""
        if self._text_enc is None:
            from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
            self._text_enc = hashed_ngram_encode(dim=self.dim)
        return _unit(self._text_enc(str(text)))

    # ---------------------------------------------------------------- encoders
    def encode_call(self, verb, args, weight=1.0):
        """VERB(x)verb + sum_k SLOT_k(x)value_k, each term unit norm, times `weight`. NOT renormalised: the norm
        (~sqrt(1 + #slots)) is information the decoder uses to set its noise scale."""
        v = bind(self.role(self.VERB), self.atom(verb, "verb"))
        for slot in sorted(args):
            v = v + bind(self.role(slot), self._filler(slot, args[slot]))
        return float(weight) * v

    def encode_soft(self, verbs, slots):
        """A state of BELIEFS rather than one call: verbs = {verb: weight}, slots = {slot: {value: weight}} ->
        sum of weighted role-filler terms. This is what a reader that is not sure produces; the decoder's chimera
        check turns split weights into an 'ambiguous' verdict instead of a confident guess."""
        v = np.zeros(self.dim)
        for verb, w in verbs.items():
            v = v + float(w) * bind(self.role(self.VERB), self.atom(verb, "verb"))
        for slot, dist in slots.items():
            R = self.role(slot)
            for value, w in dist.items():
                v = v + float(w) * bind(R, self._filler(slot, value))
        return v

    def encode_text(self, text):
        """A free-text candidate: TEXT(x)ngrams(text)."""
        return bind(self.role(self.TEXT), self.text_vec(text))

    def encode_option(self, name, examples=()):
        """A typed option: OPTION(x)name + TEXT(x)ngrams(examples bundled). Decode the name with
        decode_role(v, 'OPTION', names); compare the TEXT part to a question with option_text_similarity."""
        v = bind(self.role(self.OPTION), self.atom(name, "opt"))
        ex = [self.text_vec(e) for e in (examples or ())]
        if ex:
            v = v + bind(self.role(self.TEXT), _unit(np.sum(ex, axis=0)))
        return v

    def option_text_similarity(self, option_vec, text):
        """Cosine between the option's TEXT part (unbound exactly) and a question's n-gram vector."""
        return cosine(unbind(option_vec, self.role(self.TEXT)), self.text_vec(text))

    # ---------------------------------------------------------------- decoding
    def decode_role(self, state, role, candidates):
        """Per-role unbind + cleanup: the coefficient of every candidate filler under `role`, best first ->
        [(candidate, coefficient)]. With unit fillers and a unitary role the dot product IS the filler's weight in
        the state plus crosstalk ~ N(0, |state|^2 / D). Ties keep candidate order (stable sort)."""
        if not candidates:
            return []
        u = unbind(np.asarray(state, float), self.role(role))
        C = np.stack([self._filler(role, c) for c in candidates])
        s = C @ u
        order = np.argsort(-s, kind="stable")
        return [(candidates[int(j)], float(s[int(j)])) for j in order]

    def _matrix(self, role, candidates):
        """The stacked filler atoms of a candidate list, cached (a compose re-reads every role twice)."""
        kind = "verb" if role == self.VERB else ("opt" if role == self.OPTION else "val")
        key = (kind, tuple(candidates))
        M = self._mats.get(key)
        if M is None:
            if len(self._mats) > 512:                 # bounded: a long-lived mind sees many candidate lists
                self._mats.clear()
            M = self._mats[key] = np.stack([self._filler(role, c) for c in candidates])
        return M

    def _role_spec(self, role):
        F = self._specs.get(role)
        if F is None:
            F = self._specs[role] = _rfft(self.role(role))
        return F

    def _scores_spec(self, F_state, role, candidates):
        """_scores from a precomputed state spectrum: unbind = irfft(F_state * conj(F_role))."""
        if not candidates:
            return np.zeros(0)
        return self._matrix(role, candidates) @ _irfft(F_state * np.conj(self._role_spec(role)), n=self.dim)

    def _scores(self, state, role, candidates):
        if not candidates:
            return np.zeros(0)
        return self._matrix(role, candidates) @ unbind(state, self.role(role))

    @staticmethod
    def _assign_group(slots, cands, S, distinct):
        """Pick one candidate per slot. `S[k]` = coefficients of slot k over the SHARED candidate list `cands`.
        distinct=True: no two slots of the group take the same value (from != to) -- the joint argmax over
        injective assignments, enumerated in lexicographic order so a tie keeps the positional assignment. Large
        groups (> 20,000 assignments) fall back to greedy, strongest coefficient first."""
        k, n = len(slots), len(cands)
        if not distinct or k == 1 or n < k:
            return [int(np.argmax(S[i])) if n else None for i in range(k)]
        if k == 2:
            # the common case (from/to) as one array op: T[i, j] = S0[i] + S1[j], i != j. np.argmax on the
            # flattened matrix returns the FIRST maximum in row-major (= lexicographic) order, the same tie rule
            # as the loop below. (The loop was 14 ms of a 23 ms compose at 50 values -- profiled.)
            T = np.asarray(S[0], float)[:, None] + np.asarray(S[1], float)[None, :]
            np.fill_diagonal(T, -np.inf)
            i, j = np.unravel_index(int(np.argmax(T)), T.shape)
            return [int(i), int(j)]
        if math.perm(n, k) <= 20000:
            best, pick = -np.inf, None
            for perm in itertools.permutations(range(n), k):
                tot = sum(S[i][perm[i]] for i in range(k))
                if tot > best + 1e-12:
                    best, pick = tot, perm
            return list(pick)
        pick, used = [None] * k, set()
        flat = sorted(((float(S[i][j]), i, j) for i in range(k) for j in range(n)), key=lambda t: (-t[0], t[1], t[2]))
        for _s, i, j in flat:
            if pick[i] is None and j not in used:
                pick[i], used = j, used | {j}
        return pick

    def compose(self, state, verbs, slot_values, distinct=True, z_present=5.0, z_margin=3.0, null_m=200):
        """Decode a call from a state by per-role unbind + cleanup, then run the CHIMERA CHECK.

        verbs       : [verb, ...] or {verb: [its slot names]} (a signature limits which slots belong to the call)
        slot_values : {slot: [candidate values]}; slots whose candidate lists are IDENTICAL form a group that,
                      with distinct=True, never repeats a value (from != to)
        z_present   : a filler is 'present' when its coefficient is this many noise-sd above zero
        z_margin    : the decoded filler must beat the strongest rival left in the residual by this many sd

        Returns a CANDIDATE (never an action): {verb, args, call, verdict, score, p_model, p_null, explained,
        roles, executes: False}. verdict: 'clean' (the state holds one call), 'dominant' (a second call or a split
        belief is present but every role's pick beats its rival by z_margin), 'ambiguous' (some role's rival is
        as strong as the pick -- the chimera risk), 'empty' (no verb present). score = the weakest role's margin
        in noise sd (what a calibrator maps to p_correct). p_model = the Gaussian-crosstalk probability that
        every pick is right, times each role's share of its coefficient mass (a cold-start estimate; the mind's
        faculty replaces it with a calibrated p once outcomes exist). p_null = procedure-matched null p-value
        that a STRUCTURELESS state of this shape has every role this present (small = a real call is there)."""
        state = np.asarray(state, dtype=np.float64)
        # a slot with no candidates cannot be decoded (a question with no number has nothing for 'amount')
        slot_values = {r: list(v) for r, v in slot_values.items() if len(v)}
        sig = verbs if isinstance(verbs, dict) else {v: None for v in verbs}
        verb_list = list(sig)
        norm = float(np.linalg.norm(state))
        base = {"verb": None, "args": {}, "call": None, "verdict": "empty", "score": 0.0, "p_model": 0.0,
                "p_null": 1.0, "explained": 0.0, "roles": {}, "executes": False, "candidate": True}
        if norm == 0.0 or not verb_list:
            return base
        # The noise scale of ONE coefficient: crosstalk from every other term is ~N(0, |state|^2 / D). A unitary
        # unbind preserves the norm, so this holds for every role -- no probe bank needed (checked in _selftest).
        sigma = norm / math.sqrt(self.dim)
        roles = [self.VERB] + list(slot_values)
        cands = {self.VERB: verb_list}
        cands.update({r: list(v) for r, v in slot_values.items()})
        # ONE forward FFT of the state; every role is then unbound in the frequency domain (a unitary role's
        # involution is its conjugate spectrum), so a compose costs 2 * #roles + 5 FFTs instead of 27 (profiled:
        # FFTs were the whole remaining cost after the pair assignment was vectorised). Equal to holographic_ai's
        # unbind up to ~1e-16 -- the same order bundle_bind documents -- which can only flip an EXACT tie.
        Fs = _rfft(state)
        S = {r: self._scores_spec(Fs, r, cands[r]) for r in roles}

        # ---- 1. per-role pick (joint within a group of slots that share one candidate list)
        pick = {self.VERB: int(np.argmax(S[self.VERB]))}
        groups = {}
        for r in slot_values:
            groups.setdefault(tuple(cands[r]), []).append(r)
        for shared, members in groups.items():
            got = self._assign_group(members, list(shared), [S[r] for r in members], distinct)
            for r, j in zip(members, got):
                pick[r] = j
        verb = verb_list[pick[self.VERB]]
        if S[self.VERB][pick[self.VERB]] / sigma < z_present:
            base.update(roles={self.VERB: {"value": verb, "z": float(S[self.VERB][pick[self.VERB]] / sigma)}})
            return base
        belongs = set(slot_values) if sig.get(verb) is None else set(sig[verb]) & set(slot_values)
        present = [self.VERB] + [r for r in slot_values if r in belongs and pick[r] is not None
                                 and S[r][pick[r]] / sigma >= z_present]

        # ---- 2. recompose and EXPLAIN AWAY: subtract each decoded term at its measured coefficient
        Fc = np.zeros_like(Fs)
        for r in present:
            val = cands[r][pick[r]]
            Fc = Fc + float(S[r][pick[r]]) * self._role_spec(r) * _rfft(self._filler(r, val))
        c_hat = _irfft(Fc, n=self.dim)
        Fres = Fs - Fc
        explained = cosine(c_hat, state)
        resid = state - c_hat
        # The residual's OWN noise scale. Explaining the decoded call away also removes the crosstalk its terms
        # were putting on every other reading, so a rival left in the residual is read against |resid|/sqrt(D),
        # not |state|/sqrt(D). (Found by the selftest: a 0.3-weight rival read 3.2 sd in the state and 5.0 sd in
        # the residual against the STATE's sigma -- 5.6 expected -- and 18 sd against the residual's own.)
        sigma_res = max(float(np.linalg.norm(resid)) / math.sqrt(self.dim), 1e-12)

        # ---- 3. re-read EVERY role in the residual (a slot outside the decoded verb's signature that still holds
        # a filler means another call is superposed here -- a blend -- even if every picked role looks clean)
        info, margins, leftover_any, role_lp = {}, [], False, {}
        for r in roles:
            R = self._scores_spec(Fres, r, cands[r])
            if r in present and R.size:
                # the pick's OWN residual coefficient is its estimation error (1 - a), not a rival: it lies exactly
                # along the pick, so against sigma_res it reads as a huge 'leftover' (found by test: 12 sd on a
                # clean call). A rival is any OTHER candidate -- including a value the sibling slot picked, which
                # is how 'between X and Y' (X and Y in both roles) is caught.
                R = R.copy()
                R[pick[r]] = -np.inf
            rho = float(R.max()) if R.size and np.isfinite(R.max()) else 0.0
            left_z = rho / sigma_res
            leftover_any = leftover_any or (left_z >= z_present)
            if r not in present:
                if r in belongs and r != self.VERB:
                    info[r] = {"value": None, "z": float(S[r].max() / sigma) if S[r].size else 0.0,
                               "leftover_z": left_z}
                elif left_z >= z_present:
                    info[r] = {"value": None, "outside_signature": True, "leftover_z": left_z}
                continue
            a = float(S[r][pick[r]])
            m = (a - rho) / sigma
            margins.append(m)
            # cold-start probability that THIS role's pick is right: Gaussian crosstalk against every rival in the
            # ORIGINAL scores, times the pick's share of the role's positive coefficient mass (a 50/50 belief has
            # share 0.5 however clean the crosstalk is)
            rivals = [float(x) for j, x in enumerate(S[r]) if j != pick[r]]
            lp = sum(math.log(max(_phi((a - x) / (math.sqrt(2.0) * sigma)), 1e-300)) for x in rivals)
            # rivals count toward the mass only ABOVE a 2-sd dead zone: with a 1-sd zone the ~16% of noise rivals
            # past 1 sd cost a clean 3-role call over 50 values its confidence (p_model 0.72 -- found by test)
            mass = a + sum(max(x - 2.0 * sigma, 0.0) for x in rivals)
            share = a / mass if mass > 0 else 0.0
            role_lp[r] = lp + math.log(max(share, 1e-300))
            ranked = sorted(zip(cands[r], (float(x) / sigma for x in S[r])), key=lambda t: -t[1])[:3]
            info[r] = {"value": cands[r][pick[r]], "z": a / sigma, "leftover_z": left_z, "margin_z": m,
                       "share": share, "ranked": ranked}
        score = float(min(margins)) if margins else 0.0
        # p_model: roles are independent EXCEPT inside a distinct group (from/to share one pool), where one wrong
        # pick means both are wrong -- such a group counts once, by its weakest member, not as a product
        logp, done = 0.0, set()
        for shared, members in groups.items():
            ms_ = [r for r in members if r in role_lp]
            if distinct and len(ms_) > 1:
                logp += min(role_lp[r] for r in ms_)
                done.update(ms_)
        logp += sum(v for r, v in role_lp.items() if r not in done)
        if not leftover_any:
            verdict = "clean"
        elif score >= z_margin:
            verdict = "dominant"
        else:
            verdict = "ambiguous"
        args = {r: cands[r][pick[r]] for r in present if r != self.VERB}
        presence = float(min(info[r]["z"] for r in present))
        return {"verb": verb, "args": args, "call": call_string(verb, args), "verdict": verdict, "score": score,
                "p_model": float(math.exp(logp)), "p_null": self._p_null(presence, cands, present),
                "explained": float(explained), "roles": info, "executes": False, "candidate": True}

    def _p_null(self, presence, cands, roles, m=200):
        """p = (1 + #null >= presence) / (m + 1), where the null is the SAME statistic (the weakest coefficient,
        in noise sd, over the SAME roles the decoded call uses) on m structureless Gaussian states through the SAME
        per-role cleanup. Keyed by the SHAPE (dim, those roles' candidate-list sizes), cached: one fit per shape
        (~m x 0.1 ms per role)."""
        key = (self.dim, tuple(len(cands[r]) for r in roles))
        null = self._null_cache.get(key)
        if null is None:
            rng = np.random.default_rng(self.seed + 104729)
            null = np.empty(m)
            mats = {r: np.stack([self._filler(r, c) for c in cands[r]]) for r in roles}
            for i in range(m):
                s = rng.standard_normal(self.dim)
                sg = float(np.linalg.norm(s)) / math.sqrt(self.dim)
                null[i] = min(float((mats[r] @ unbind(s, self.role(r))).max()) / sg for r in roles)
            null = np.sort(null)
            self._null_cache[key] = null
        return float((1 + int((null >= presence).sum())) / (len(null) + 1))


def naive_decode(codec, state, verbs, slot_values):
    """The BASELINE the chimera check is measured against: per-role argmax, nothing else (what the panel called a
    blended readout's decode). -> {verb, args}."""
    verb = codec.decode_role(state, codec.VERB, list(verbs))[0][0]
    return {"verb": verb, "args": {r: codec.decode_role(state, r, list(v))[0][0] for r, v in slot_values.items()}}


# ============================================================================================ the product form
def bipolar_atom(seed, name, dim):
    """A deterministic +/-1 atom (MAP binding needs a self-inverse algebra; see holographic_resonator)."""
    return np.where(derived_atom(seed, name, dim) >= 0, 1.0, -1.0)


class ProductCodec:
    """A call as ONE bound product with PERMUTATION roles: verb * rho^1(x_1) * rho^2(x_2) * ... (MAP, bipolar).

    Used only where a product is what you have (a single bound key, a hashed signature) -- the sum form above is
    the default because it decodes in one unbind per role. Decoding a product is a SEARCH: the resonator
    (holographic_resonator.ResonatorNetwork) with its noise-tolerant exit and a procedure-matched p-value."""

    def __init__(self, verbs, values, slots, dim=DIM, seed=SEED):
        self.verbs, self.values, self.slots = list(verbs), list(values), list(slots)
        self.dim, self.seed = int(dim), int(seed)
        self.V = np.stack([bipolar_atom(seed, "verb:" + v, self.dim) for v in self.verbs])
        X = np.stack([bipolar_atom(seed, "val:" + x, self.dim) for x in self.values])
        # slot k's codebook is the value codebook rolled by k+1: a permutation role (rho^(k+1)), the form the panel
        # measured at 30/30 order-correct where bound role atoms were at chance
        self.books = [self.V] + [np.roll(X, k + 1, axis=1) for k in range(len(self.slots))]

    def encode(self, verb, args):
        """verb * rho^1(args[slot_1]) * rho^2(args[slot_2]) * ... -> a bipolar vector."""
        v = self.V[self.verbs.index(verb)].copy()
        for k, slot in enumerate(self.slots):
            v = v * self.books[k + 1][self.values.index(args[slot])]
        return v

    def factor(self, composite, restarts=20, iters=200, patience=8, accept_p=0.01, m_null=100):
        """Search the product with the resonator's NOISE-TOLERANT exit -> a CANDIDATE {verb, args, call, p_value,
        agreement, accepted, exit, executes: False}. p_value is the resonator's procedure-matched null p;
        'accepted' means p_value <= accept_p. Never executes anything."""
        from holographic.misc.holographic_resonator import ResonatorNetwork
        net = ResonatorNetwork(self.books)
        r = net.factor(composite, restarts=restarts, iters=iters, tolerant=True, patience=patience,
                       accept_p=accept_p, m_null=m_null)
        f = [int(i) for i in r["factors"]]
        args = {slot: self.values[f[k + 1]] for k, slot in enumerate(self.slots)}
        verb = self.verbs[f[0]]
        return {"verb": verb, "args": args, "call": call_string(verb, args), "p_value": r["p_value"],
                "agreement": r["agreement"], "accepted": r["accepted"], "exit": r["exit"],
                "restarts": r["restarts"], "executes": False, "candidate": True}


# ============================================================================================ direction, question side
class ContextRoles:
    """Per-slot-NAME prototypes of the CONTEXT a filler appears in, learned from `from_question` verdicts.

    A mention's context is a role-filler record of the words around it -- L1 (the word before), L2 (two before),
    R1 (the word after) -- each word bound to its own UNITARY context role, so 'for 100 dollars' (L1 = <num>) and
    'dollars for' (R1 = for) are different records even though the words are the same. Two masks make it
    generalise: a number is <num> (the amount, not its value, is the cue), and a word inside ANOTHER mention is
    <m> ('usd/jpy': the base currency is the one FOLLOWED by a currency). Slots are pooled by NAME, so 'from'
    learned on one verb helps every verb with a 'from' slot (the meaning index pools slot VALUES the same way).

    rule='centroid' (the panel's form: a slot's prototype is the unit mean of its contexts) or 'infonce' (the
    shared contrastive rule, holographic_protostore.ProtoStore.update: pull the truth, push the rival slot)."""

    WINDOW = ("L1", "L2", "R1")

    def __init__(self, dim=DIM, seed=SEED, window=WINDOW, mask_mentions=True, rule="centroid", lr=0.3, tau=0.05):
        if rule not in ("centroid", "infonce"):
            raise ValueError("rule must be 'centroid' or 'infonce', got %r" % (rule,))
        self.dim, self.seed = int(dim), int(seed)
        self.window, self.mask_mentions, self.rule = tuple(window), bool(mask_mentions), rule
        self.lr, self.tau = float(lr), float(tau)
        self._roles = {w: derived_atom(self.seed, "ctx:" + w, self.dim, unitary=True) for w in self.window}
        self._words = {}
        self.sums = {}                  # centroid rule: slot -> summed unit context vectors
        self.counts = {}
        self.store = None
        if rule == "infonce":
            from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
            self.store = ProtoStore(self.dim, tau=self.tau, lr=self.lr, name="context-roles")

    # ---------------------------------------------------------------- the context record
    def _word(self, tok):
        v = self._words.get(tok)
        if v is None:
            v = self._words[tok] = derived_atom(self.seed, "w:" + tok, self.dim)
        return v

    def _tok_at(self, tokens, i, others):
        if i < 0:
            return "<s>"
        if i >= len(tokens):
            return "</s>"
        if self.mask_mentions and any(a <= i < b for a, b in others):
            return "<m>"
        t = tokens[i]
        return "<num>" if t[0].isdigit() else t

    def context(self, tokens, span, others=()):
        """The unit context record of the mention at token span (a, b): sum over the window of
        bind(ctx-role, word atom). `others` = the spans of the other mentions (masked as <m>)."""
        a, b = span
        pos = {"L1": a - 1, "L2": a - 2, "L3": a - 3, "R1": b, "R2": b + 1}
        v = np.zeros(self.dim)
        for w in self.window:
            v = v + bind(self._roles[w], self._word(self._tok_at(tokens, pos[w], others)))
        return _unit(v)

    def _spans(self, tokens, mentions):
        """mentions: [words | (words, token_index)] -> spans in the given order, none overlapping."""
        spans = []
        for m in mentions:
            if isinstance(m, (tuple, list)):
                w, at = m
                n = len(question_tokens(w))
                sp = (int(at), int(at) + n) if question_tokens(w) == tokens[int(at):int(at) + n] else None
            else:
                sp = find_span(tokens, m, spans)
            if sp is None:
                raise ValueError("mention %r not found in %r" % (m, tokens))
            spans.append(sp)
        return spans

    # ---------------------------------------------------------------- learning
    def learn(self, question, from_question):
        """One `from_question` verdict: {slot: words} (or {slot: (words, token_index)} to pin an occurrence, or
        {slot: [words, ...]} for several mentions of one slot). Each mention's context moves that slot's
        prototype. Returns {'learned': [slots], 'n': contexts written}."""
        toks = question_tokens(question)
        flat = []
        for slot, w in from_question.items():
            if isinstance(w, list):
                flat.extend((slot, x) for x in w)
            else:
                flat.append((slot, w))
        spans = self._spans(toks, [w for _, w in flat])
        n = 0
        for i, (slot, _w) in enumerate(flat):
            ctx = self.context(toks, spans[i], [s for j, s in enumerate(spans) if j != i])
            if self.rule == "centroid":
                self.sums[slot] = self.sums.get(slot, np.zeros(self.dim)) + ctx
            else:
                self.store.update(ctx, slot)
            self.counts[slot] = self.counts.get(slot, 0) + 1
            n += 1
        return {"learned": sorted({s for s, _ in flat}), "n": n}

    def prototype(self, slot):
        """The slot's unit context prototype, or None if it was never taught."""
        if self.rule == "centroid":
            s = self.sums.get(slot)
            return None if s is None else _unit(s)
        if slot not in self.store:
            return None
        return self.store.P[self.store.index(slot)].copy()

    # ---------------------------------------------------------------- reading
    def read(self, question, mentions, slots, tau=None, top=6, none_slot=NONE_SLOT):
        """Assign `slots` (e.g. ['from', 'to']) to `mentions` (the words in the question that name slot values)
        by their contexts. Returns {'assignment': {slot: words}, 'order': [mention index per slot], 'margin'
        (best - second assignment, cosine units), 'p' (softmax of the hypothesis totals at tau: the reader's
        belief in the best assignment), 'ranked': [(assignment, total, p)], 'scores': cosine matrix
        (mention x slot), 'direction': bool, 'p_none', 'via': 'context' | 'positional'}.

        NO-DIRECTION HYPOTHESIS: when the `none_slot` ('either') prototype has been taught -- a from_question
        verdict {'either': [words, words]} for a sentence that names the values but no direction ('the rate
        between usd and cad') -- its total over the best assignment's mentions competes in the same softmax.
        'direction' is False when it wins; 'p_none' is its share. KEPT NEGATIVE that made this necessary: taught
        only directional verdicts, the reader was CONFIDENTLY directional on 41/41 symmetric sentences (mean
        p 0.99): a context window always finds some cue, so 'no direction' has to be learned, not inferred.

        POSITIONAL FALLBACK: a slot never taught has no prototype, so every slot is filled in mention order (the
        first mention to the first slot) -- what MeaningIndex.bind does today -- and 'via' says so, with p None."""
        toks = question_tokens(question)
        spans = self._spans(toks, mentions)
        n, k = len(mentions), len(slots)
        words = [m[0] if isinstance(m, (tuple, list)) else m for m in mentions]
        protos = [self.prototype(s) for s in slots]
        if n < k:
            raise ValueError("%d mentions cannot fill %d slots" % (n, k))
        if any(p is None for p in protos):
            asg = {slots[i]: words[i] for i in range(k)}
            return {"assignment": asg, "order": list(range(k)), "margin": 0.0, "p": None, "p_none": None,
                    "direction": True, "ranked": [(asg, 0.0, None)], "scores": None, "via": "positional"}
        ctx = [self.context(toks, spans[i], [s for j, s in enumerate(spans) if j != i]) for i in range(n)]
        S = np.array([[float(np.dot(c, p)) for p in protos] for c in ctx])
        # every injective assignment of slots to mentions, in lexicographic order of mention indices -- (0, 1)
        # before (1, 0) -- so a TIE keeps the positional reading (the stated tie rule)
        perms = list(itertools.permutations(range(n), k))
        tot = np.array([sum(S[perm[i], i] for i in range(k)) for perm in perms])
        order = np.argsort(-tot, kind="stable")
        best = perms[int(order[0])]
        # the no-direction hypothesis, scored over the same mentions the best assignment uses
        pn = self.prototype(none_slot) if (none_slot and k >= 2) else None
        t_none = None if pn is None else float(sum(np.dot(ctx[i], pn) for i in best))
        allt = np.append(tot, t_none) if t_none is not None else tot
        t = float(self.tau if tau is None else tau)
        z = (allt - allt.max()) / max(t, 1e-9)
        pr = np.exp(z) / np.exp(z).sum()
        margin = float(tot[order[0]] - tot[order[1]]) if len(order) > 1 else float("inf")
        ranked = [({slots[i]: words[perms[int(j)][i]] for i in range(k)}, float(tot[int(j)]), float(pr[int(j)]))
                  for j in order[:top]]
        return {"assignment": {slots[i]: words[best[i]] for i in range(k)}, "order": list(best),
                "margin": margin, "p": float(pr[int(order[0])]),
                "p_none": None if t_none is None else float(pr[-1]),
                "direction": t_none is None or bool(tot[order[0]] >= t_none),
                "ranked": ranked, "scores": S.tolist(), "via": "context"}

    # ---------------------------------------------------------------- persistence
    def state(self):
        """(meta, arrays) -- the container section shape, so the learned prototypes can live in a partition."""
        meta = {"dim": self.dim, "seed": self.seed, "window": list(self.window), "mask": self.mask_mentions,
                "rule": self.rule, "lr": self.lr, "tau": self.tau, "counts": dict(self.counts),
                "slots": sorted(self.sums)}
        arrays = {"sums": np.stack([self.sums[s] for s in meta["slots"]]) if self.sums
                  else np.zeros((0, self.dim))}
        if self.store is not None:
            smeta, sarr = self.store.state()
            meta["store"] = smeta
            arrays["store_A"] = sarr["A"]
        return meta, arrays

    @classmethod
    def from_state(cls, meta, arrays):
        """Rebuild from state()."""
        cr = cls(meta["dim"], meta["seed"], tuple(meta["window"]), meta["mask"], meta["rule"], meta["lr"],
                 meta["tau"])
        cr.counts = dict(meta.get("counts", {}))
        for i, s in enumerate(meta.get("slots", [])):
            cr.sums[s] = np.asarray(arrays["sums"][i], dtype=np.float64)
        if meta.get("store") is not None:
            from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
            cr.store = ProtoStore.from_state(meta["store"], {"A": arrays["store_A"]})
        return cr

    def digest(self):
        """sha256 of the learned state -- two runs, one digest (the determinism receipt)."""
        meta, arr = self.state()
        h = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode())
        for k in sorted(arr):
            h.update(np.ascontiguousarray(np.asarray(arr[k], np.float64)).tobytes())
        return h.hexdigest()


def positional_read(mentions, slots):
    """The BASELINE: fill slots in the order the mentions appear (first currency named = FROM)."""
    words = [m[0] if isinstance(m, (tuple, list)) else m for m in mentions]
    return {slots[i]: words[i] for i in range(len(slots))}


def question_state(codec, reader, question, verb, mentions, slots, numbers=None, number_slot="amount", tau=None):
    """Build the SOFT call state of a question: the reader's belief over slot assignments becomes the weights.

        state = VERB(x)verb + sum over assignments pi of P(pi) * sum_k SLOT_k(x)value(mention pi_k)
                + number_slot(x)num   (each number found, weight 1/#numbers)

    mentions = {words: value} (the question's words that name known values, in any order -- they are located in
    the question); slots = the choice slots they fill. The state is the reader's EXPECTED call: a confident
    reader gives one clean call, a split reader gives a blend -- which compose() then reports as 'ambiguous'
    instead of picking a side. Returns (state, reading)."""
    toks = question_tokens(question)
    words = list(mentions)
    spans = []
    for w in words:
        sp = find_span(toks, w, spans)
        if sp is None:
            raise ValueError("mention %r not in question %r" % (w, question))
        spans.append(sp)
    ordered = [w for _, w in sorted(zip(spans, words))]           # mention order in the question
    rd = reader.read(question, ordered, slots, tau=tau, top=10 ** 6) if reader is not None else None
    dist = {s: {} for s in slots}
    if rd is None or rd["via"] == "positional":
        asg = positional_read(ordered, slots)
        for s in slots:
            dist[s][mentions[asg[s]]] = 1.0
    else:
        # the no-direction mass (p_none) means "either order": spread it evenly over every assignment, which
        # turns the state into a blend -- exactly what compose()'s chimera check reports as 'ambiguous'
        spread = (rd["p_none"] or 0.0) / max(len(rd["ranked"]), 1)
        for asg, _tot, p in rd["ranked"]:
            for s in slots:
                v = mentions[asg[s]]
                dist[s][v] = dist[s].get(v, 0.0) + p + spread
    nums = amounts_in(question) if numbers is None else list(numbers)
    if nums and number_slot:
        dist[number_slot] = {}
        for x in nums:
            dist[number_slot][x] = dist[number_slot].get(x, 0.0) + 1.0 / len(nums)
    return codec.encode_soft({verb: 1.0}, dist), rd


# ============================================================================================ trajectories (E5.1)
def trajectory_encode(step_vectors):
    """sum_t rho^t(step_t), unit-normalised: a permutation-ordered bundle. Each step is unit-normalised first so
    no step outweighs another; rho^t is holographic_ai.permute (a cyclic shift -- exact, orthogonal, reversible).
    Step vectors come from RoleCodec.encode_call / encode_step, or RecordCodec.encode(DecisionRecord).

    MEASURED (bench_rolecall.json 'trajectory', D=2048, 200 trials per T; the NOISE FLOOR is the same plan
    re-encoded with every step's incidental text changed, and 'detected' means below the floor's 1st percentile):
      text_weight 0.5, T = 5 / 10 / 20: adjacent swaps 99.5 / 96.5 / 98.0%, one step inserted anywhere
        100 / 100 / 99.5%, inserted at the END 100 / 100 / 96.5%, deleted 100%, one argument changed 100 / 100 / 91.5%.
      text_weight 1.0 (incidental text as heavy as the call): swaps 97 / 90.5 / 86.5%, insert anywhere 94.5 / 95 /
        96.5%, deleted 94-95%. The draft's permutation-only gate passed trivially; insertions are the harder test.
      Decode-and-diff (trajectory_decode + trajectory_diff) named the edit 37-40 of 40 in every cell.
    KEPT NEGATIVE, loud: with text_weight 1.0 an insertion at the very END of a long trajectory is mostly INSIDE
    the noise floor -- detected 64.5% (T=5), 23.5% (T=10), 14.5% (T=20): appending one step moves the cosine by
    ~sqrt(T/(T+1)), which heavy incidental text swamps. A changed argument is likewise 51 / 16.5 / 16% there. Keep
    incidental text light (text_weight <= 0.5) or compare by decode-and-diff, which names the append exactly."""
    vs = [_unit(v) for v in step_vectors]
    if not vs:
        raise ValueError("a trajectory needs at least one step")
    return _unit(np.sum([permute(v, t) for t, v in enumerate(vs)], axis=0))


def encode_step(codec, verb, args, text=None, text_weight=1.0):
    """One trajectory step as a role-filler record: the call (VERB + slots) plus, optionally, TEXT(x)ngrams(text)
    for the step's incidental text (a state line, a log excerpt). text_weight sets how much incidental text may
    move the step: it is the 'noise' a same-plan re-run carries, so a caller comparing PLANS keeps it low."""
    v = codec.encode_call(verb, args)
    if text:
        v = v + float(text_weight) * codec.encode_text(text)
    return v


def trajectory_decode(vec, step_book, max_len=64, z_present=5.0):
    """Read the steps back position by position: unpermute by t, clean up against `step_book` ({name: step
    vector}); stop at the first position whose best step is not z_present noise-sd above zero. -> [(name, z)]."""
    names = list(step_book)
    if not names:
        return []
    M = np.stack([_unit(step_book[n]) for n in names])
    vec = np.asarray(vec, float)
    sigma = float(np.linalg.norm(vec)) / math.sqrt(len(vec))
    out = []
    for t in range(int(max_len)):
        s = M @ permute(vec, -t)
        j = int(np.argmax(s))
        if sigma <= 0 or s[j] / sigma < z_present:
            break
        out.append((names[j], float(s[j] / sigma)))
    return out


def trajectory_diff(a_steps, b_steps):
    """Explain how two decoded step sequences differ (difflib opcodes, stdlib): [{'op': 'replace' | 'delete' |
    'insert', 'a': [...], 'b': [...], 'at': index in a}]; an adjacent swap shows as its own op 'swap'."""
    import difflib
    a, b = list(a_steps), list(b_steps)
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        ops.append({"op": tag, "a": a[i1:i2], "b": b[j1:j2], "at": i1})
    # an adjacent swap reads as delete + insert (or two replaces); name it when it is exactly that
    if len(a) == len(b) and len(ops) >= 1:
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return [{"op": "swap", "a": a[diff[0]:diff[1] + 1], "b": b[diff[0]:diff[1] + 1], "at": diff[0]}]
    return ops


def trajectory_compare(a, b, floor=None):
    """Cosine between two trajectory vectors, and -- when `floor` (cosines of SAME-plan re-encodings, the noise
    floor) is given -- p_same = (1 + #floor <= cosine) / (n + 1): the chance a same-plan pair looks this
    DIFFERENT. Small p_same -> an edit (swap / insert / delete) is present above the noise floor."""
    c = cosine(a, b)
    out = {"cosine": float(c)}
    if floor is not None and len(floor):
        f = np.asarray(floor, float)
        out["p_same"] = float((1 + int((f <= c).sum())) / (len(f) + 1))
        out["floor_p01"] = float(np.quantile(f, 0.01))
    return out


# ============================================================================================ calibration metric
def expected_calibration_error(p, correct, bins=10):
    """ECE with equal-width bins: sum_b (n_b / n) * |mean(p_b) - mean(correct_b)|. Returns (ece, table) where the
    table lists (lo, hi, n, mean_p, accuracy) for non-empty bins."""
    p = np.asarray(p, float)
    y = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    e, table = 0.0, []
    for i in range(int(bins)):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if sel.any():
            mp, acc = float(p[sel].mean()), float(y[sel].mean())
            e += sel.mean() * abs(mp - acc)
            table.append((float(lo), float(hi), int(sel.sum()), mp, acc))
    return float(e), table


# ============================================================================================ selftest
def _selftest():
    """Small, fast pins of every claim above (the full measurement is tools/bench_rolecall.py)."""
    D = 1024
    codec = RoleCodec(dim=D, seed=0)
    verbs = ["fx", "weather", "price", "timer"]
    vals = ["USD", "EUR", "JPY", "GBP", "CAD", "MXN"]
    slots = {"from": vals, "to": vals}
    # 1. a clean sum decodes exactly, verdict clean, never executes
    s = codec.encode_call("fx", {"from": "EUR", "to": "JPY"})
    got = codec.compose(s, {"fx": ["from", "to"], "weather": [], "price": [], "timer": []}, slots)
    assert got["call"] == "fx(from=EUR, to=JPY)" and got["verdict"] == "clean", got
    assert got["executes"] is False and got["p_null"] < 0.05
    # 2. the noise scale: |state| / sqrt(D) matches the empirical sd of non-member coefficients
    sd = np.std([float(np.dot(codec.atom("junk%d" % i), unbind(s, codec.role("from")))) for i in range(200)])
    assert 0.7 < sd / (np.linalg.norm(s) / math.sqrt(D)) < 1.3, sd
    # 3. an equal blend of two calls is flagged ambiguous, never passed off as one call
    blend = s + codec.encode_call("fx", {"from": "USD", "to": "GBP"})
    assert codec.compose(blend, ["fx"], slots)["verdict"] == "ambiguous"
    # ...while a dominant blend decodes the dominant call
    dom = codec.compose(s + 0.3 * codec.encode_call("fx", {"from": "USD", "to": "GBP"}), ["fx"], slots)
    assert dom["call"] == "fx(from=EUR, to=JPY)" and dom["verdict"] == "dominant", dom
    # 4. a structureless state has no verb present
    rnd = np.random.default_rng(1).standard_normal(D)
    assert codec.compose(rnd, verbs, slots)["verdict"] == "empty"
    # 5. options: the name decodes and the TEXT part matches its own examples best
    ov = codec.encode_option("refund", ["i want my money back", "refund my order"])
    assert codec.decode_role(ov, "OPTION", ["refund", "shipping"])[0][0] == "refund"
    assert codec.option_text_similarity(ov, "money back please") > codec.option_text_similarity(ov, "zzz qqq")
    # 6. direction from context: learn two phrasings, read a third whose TO currency comes first
    cr = ContextRoles(dim=D)
    cr.learn("how many pesos can i get for 20 dollars", {"from": "dollars", "to": "pesos"})
    cr.learn("how many yen can i get for 5 euros", {"from": "euros", "to": "yen"})
    cr.learn("convert 10 pounds to euros", {"from": "pounds", "to": "euros"})
    r = cr.read("how many rupees can i get for 50 pounds", ["rupees", "pounds"], ["from", "to"])
    assert r["assignment"] == {"from": "pounds", "to": "rupees"} and r["via"] == "context", r
    assert positional_read(["rupees", "pounds"], ["from", "to"])["from"] == "rupees"     # the baseline is wrong
    assert ContextRoles.from_state(*cr.state()).digest() == cr.digest()
    # 7. the soft state of that question composes the right call; an untaught reader falls back to positional
    st, _ = question_state(codec, cr, "how many rupees can i get for 50 pounds", "fx",
                           {"rupees": "INR", "pounds": "GBP"}, ["from", "to"])
    c = codec.compose(st, {"fx": ["from", "to", "amount"]}, {"from": ["INR", "GBP"], "to": ["INR", "GBP"],
                                                              "amount": ["50"]})
    assert c["args"] == {"from": "GBP", "to": "INR", "amount": "50"}, c
    assert ContextRoles(dim=D).read("usd to eur", ["usd", "eur"], ["from", "to"])["via"] == "positional"
    # 8. trajectories: order and an inserted step both move the vector; the steps decode back in order
    steps = [codec.encode_call("fx", {"from": a, "to": b}) for a, b in [("USD", "EUR"), ("EUR", "JPY"),
                                                                        ("JPY", "GBP"), ("GBP", "CAD")]]
    A = trajectory_encode(steps)
    assert cosine(A, trajectory_encode([steps[1], steps[0]] + steps[2:])) < 0.8
    assert cosine(A, trajectory_encode(steps[:2] + [codec.encode_call("timer", {})] + steps[2:])) < 0.8
    book = {"s%d" % i: v for i, v in enumerate(steps)}
    assert [n for n, _ in trajectory_decode(A, book)] == ["s0", "s1", "s2", "s3"]
    assert trajectory_diff(["a", "b", "c"], ["b", "a", "c"])[0]["op"] == "swap"
    # 9. numbers
    assert amounts_in("Convert 1,250.00 EUR") == ["1250.00"] and amounts_in("10k yen to usd") == ["10000"]
    assert amounts_in("2.5 million yen") == ["2500000"] and amounts_in("five dollars") == ["5"]
    print("OK: holographic_rolecall self-test passed (sum-form decode + chimera check: clean/dominant/ambiguous; "
          "context-role direction read a TO-first question the positional rule gets wrong; trajectories separate "
          "swaps and insertions and decode back in order)")


if __name__ == "__main__":
    _selftest()
