"""A context-conditioned word generator: the honest answer to 'why isn't the
brain an LLM, and how close can deeper conditioning get?'

The character n-gram (holographic_text.HolographicNGram) generates from
P(next char | last few chars) -- shallow conditioning, good local texture, no
sense of topic. An LLM's power is that its P(next | context) is conditioned on
the whole prior context through a high-capacity LEARNED function. We cannot grow
that function on this substrate, but we CAN deepen the conditioning a little:
generate at the WORD level, where a word n-gram gives local fluency, and re-rank
its candidates by how well each candidate's learned MEANING vector aligns with a
running TOPIC vector (a decaying bundle of the content words generated so far,
seeded from the prompt). A tunable topic_weight blends the two:

    score(next) = log P_ngram(next | recent words)  +  topic_weight * cos(meaning(next), topic)

topic_weight = 0 is exactly the bare word n-gram (the baseline). Turning it up
pulls generation toward the topic.

THE HONEST TRADEOFF, AS ACTUALLY MEASURED (kept whichever way it fell -- and it
fell toward a negative, which is the valuable part). Sweeping topic_weight on a
multi-category Brown model and tracking three metrics:
  * topic_coherence: cosine of the continuation's content-word bundle to the
    seed's topic vector -- are we still about the same thing?
  * transition_validity: fraction of adjacent output pairs seen in training -- a
    grammaticality proxy the bare n-gram maximises by construction.
  * lexical diversity: unique / total words -- the guard that catches degeneracy.

The result: topic-pull re-ranking does NOT buy genuine coherence on this
substrate. At a word n-gram of order 2, ~85% of contexts have exactly ONE
continuation, so the topic term has nothing to choose among and the curve is
flat. Dropping to order 1 gives it room (~4 candidates/context), but then the
honest failure shows: moderate topic_weight slightly LOWERS coherence, and only
extreme weight raises the coherence NUMBER -- while lexical diversity collapses
(0.78 -> 0.09) into degenerate repetition ('have , have , said , has , have'),
which even keeps transition_validity high because 'have have' is a seen bigram.
The high-weight 'coherence' is the metric being gamed by a topic vector
collapsing onto a few high-frequency words, not real on-topic language.

WHY THIS IS THE EXPECTED, INSTRUCTIVE OUTCOME. It confirms, by measurement, the
explanation for why the brain is not an LLM: the missing piece is not the loop or
the re-ranking -- it is a high-capacity LEARNED P(next | context). A shallow word
n-gram proposes candidates with no deep structure, so re-ranking them by a
bag-of-meanings topic vector cannot conjure coherence that the proposer never
had. You can only re-rank structure that is already present in the candidates.
Deepening the conditioning with a topic bundle is too weak a lever; the honest
ceiling is set by the proposer, exactly as argued. The negative is the finding.

(topic_weight = 0 recovers the bare word n-gram baseline, so the class doubles as
a clean word-level generator; the value here is the measurement apparatus and the
kept negative, not a coherence win.)

Needs: numpy, holographic_text, holographic_encoders.
"""
import math
from collections import Counter, defaultdict

import numpy as np

from holographic_encoders import TextEncoder
from holographic_text import _content, _tokens, STOPWORDS
from holographic_ai import cosine, bundle


class ContextGenerator:
    """Word n-gram (local fluency) re-ranked by topic alignment (global
    coherence), with a tunable topic_weight. topic_weight=0 is the bare n-gram."""

    def __init__(self, dim=512, order=2, window=2, seed=0, decay=0.85):
        self.dim = dim
        self.order = order               # word n-gram order (context length in words)
        self.decay = decay               # how fast the running topic vector forgets
        self.enc = TextEncoder(dim, window=window, seed=seed)
        self._trans = defaultdict(Counter)   # (w_{i-order..i-1}) -> Counter(next word)
        self._bigrams = set()                # (a, b) pairs seen, for transition_validity
        self._vocab = set()

    def fit(self, sentences):
        """Learn word meaning vectors (co-occurrence) and word-transition counts
        from a list of sentences (strings or token lists)."""
        for s in sentences:
            toks = s if isinstance(s, list) else _tokens(s)
            if not toks:
                continue
            self._vocab.update(toks)
            self.enc.learn([w for w in toks if w not in STOPWORDS] or toks)
            for i in range(len(toks)):
                if i > 0:
                    self._bigrams.add((toks[i - 1], toks[i]))
                ctx = tuple(toks[max(0, i - self.order):i])
                self._trans[ctx][toks[i]] += 1
        return self

    def _candidates(self, recent):
        """Next-word counts after the recent-word context, backing off to shorter
        contexts when the full one was never seen (the n-gram backoff)."""
        for k in range(min(self.order, len(recent)), -1, -1):
            ctx = tuple(recent[len(recent) - k:]) if k else ()
            if ctx in self._trans and self._trans[ctx]:
                return self._trans[ctx]
        return None

    def generate(self, seed, length=40, topic_weight=0.0, temperature=0.7, seed_rng=0):
        """Generate `length` words. topic_weight blends topic alignment into the
        n-gram choice (0 = pure n-gram baseline). Returns the generated token list
        (not including the seed)."""
        rng = np.random.default_rng(seed_rng)
        seed_toks = seed if isinstance(seed, list) else _tokens(seed)
        recent = list(seed_toks)
        # topic vector: decaying bundle of content-word meanings, seeded from prompt
        topic = bundle([self.enc.wordvec(w) for w in seed_toks
                        if w not in STOPWORDS]) if seed_toks else np.zeros(self.dim)
        out = []
        for _ in range(length):
            cand = self._candidates(recent)
            if not cand:
                break
            words = list(cand)
            counts = np.array([cand[w] for w in words], float)
            logp = np.log(counts / counts.sum())                     # n-gram log-prob
            if topic_weight and np.linalg.norm(topic) > 0:
                align = np.array([cosine(self.enc.wordvec(w), topic) for w in words])
                score = logp + topic_weight * align
            else:
                score = logp
            # temperature sample over the blended score
            w = score / max(temperature, 1e-6)
            w = np.exp(w - w.max())
            choice = words[int(rng.choice(len(words), p=w / w.sum()))]
            out.append(choice)
            recent.append(choice)
            if choice not in STOPWORDS:                              # update topic
                topic = self.decay * topic + self.enc.wordvec(choice)
        return out

    # ---- honest metrics -------------------------------------------------
    def topic_vector(self, seed):
        toks = seed if isinstance(seed, list) else _tokens(seed)
        cw = [self.enc.wordvec(w) for w in toks if w not in STOPWORDS]
        return bundle(cw) if cw else np.zeros(self.dim)

    def topic_coherence(self, tokens, topic):
        """Cosine of the continuation's content-word bundle to the seed topic."""
        cw = [self.enc.wordvec(w) for w in tokens if w not in STOPWORDS]
        if not cw or np.linalg.norm(topic) == 0:
            return 0.0
        return float(cosine(bundle(cw), topic))

    def transition_validity(self, tokens):
        """Fraction of adjacent output pairs that were seen in training -- a
        grammaticality proxy the bare n-gram maximises by construction."""
        if len(tokens) < 2:
            return 0.0
        ok = sum((tokens[i], tokens[i + 1]) in self._bigrams for i in range(len(tokens) - 1))
        return ok / (len(tokens) - 1)

    @staticmethod
    def diversity(tokens):
        """Unique / total words -- the guard that exposes degenerate repetition,
        which a topic vector collapsing onto a few frequent words produces and
        which the coherence number alone would hide."""
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    def sweep(self, seeds, weights=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0), length=40, reps=5):
        """Run the honest tradeoff sweep: for each topic_weight, mean coherence,
        transition validity, and diversity over seeds x reps. Returns a list of
        dicts. Reading it: coherence that 'rises' as diversity collapses is the
        metric being gamed, not real on-topic language."""
        rows = []
        for tw in weights:
            cohs, tvs, divs = [], [], []
            for s in seeds:
                topic = self.topic_vector(s)
                for r in range(reps):
                    toks = self.generate(s, length=length, topic_weight=tw, seed_rng=r)
                    cohs.append(self.topic_coherence(toks, topic))
                    tvs.append(self.transition_validity(toks))
                    divs.append(self.diversity(toks))
            rows.append({"topic_weight": tw,
                         "coherence": float(np.mean(cohs)),
                         "transition_validity": float(np.mean(tvs)),
                         "diversity": float(np.mean(divs))})
        return rows
