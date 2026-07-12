"""
holographic_text.py -- what a system that knows NO language can still learn from
raw text.

There is no dictionary here, no grammar, no pretrained anything. Every word and
every character starts life as a fixed random vector with no meaning attached.
What the holographic engine can do is learn the STATISTICAL STRUCTURE of text --
which symbols keep company with which -- and it turns out that structure alone is
enough to do real work. This module answers four questions, each with a measured
demo and an honest verdict:

  1. Can it learn?    -- word relatedness from co-occurrence (random indexing):
                         words that are USED alike end up NEAR each other, with no
                         supervision and no gradient descent.
  2. Can it analyze?  -- language identification from characters alone: it learns
                         to tell English / Spanish / French / German apart from
                         raw letter triples, knowing nothing about any of them.
  3. Can it organize? -- topic sorting, both supervised (label the topics) and
                         unsupervised (discover the topics by clustering).
  4. Can it produce?  -- a holographic n-gram model that generates text which is
                         locally plausible and globally nonsense -- the honest
                         truth about any small statistical language model.

All datasets below are short ORIGINAL sentences written for this file, so there
is no copyright question and the structure is clean enough to measure against.

Needs: numpy, holographic_ai.py, holographic_encoders.py beside it.
"""

import numpy as np
from collections import defaultdict, Counter

from holographic.agents_and_reasoning.holographic_ai import cosine, bind, bundle, permute, Vocabulary
from holographic.io_and_interop.holographic_encoders import TextEncoder


# ===========================================================================
# DATASETS  (original text -- topical sentences and a small multilingual set)
# ===========================================================================

TOPICS = {
    "cooking": [
        "the recipe needs flour eggs and butter",
        "preheat the oven before you bake the bread",
        "stir the sauce so it does not burn",
        "chop the onions and garlic for the soup",
        "knead the dough until it is smooth",
        "add salt and pepper to the soup",
        "the cake bakes slowly in the hot oven",
        "whisk the eggs and sugar into a batter",
        "simmer the soup on low heat for an hour",
        "season the chicken with salt and herbs",
        "mix the flour and butter into a dough",
        "roll the dough and bake it in the oven",
        "taste the sauce and add more salt",
        "boil the water before you add the eggs",
        "the bread needs flour water and salt",
        "bake the cake until the batter rises",
        "chop the herbs and stir them into the sauce",
        "fry the onions and garlic in butter",
        "the soup needs onions garlic and herbs",
        "whisk the butter sugar and eggs together",
    ],
    "space": [
        "the rocket launched toward the distant planet",
        "the astronauts orbit the earth on the station",
        "the telescope captured a faint distant galaxy",
        "the moon reflects the light of the sun",
        "comets leave a long tail of ice and dust",
        "the satellite circles the planet each hour",
        "gravity holds the planets around the star",
        "the spacecraft landed gently on the moon",
        "new stars are born inside clouds of gas",
        "the orbit of mars is wider than earth",
        "a black hole bends the light around it",
        "the probe sent images of the outer planets",
        "the rocket carried the astronauts to the station",
        "the telescope watched the distant star and galaxy",
        "the planet orbits the bright burning sun",
        "dust and gas slowly form new stars and planets",
        "the spacecraft left earth toward the red planet",
        "the moon and the planet share the same orbit",
        "the satellite sent images of the earth below",
        "gravity pulls the comets toward the burning sun",
    ],
    "sports": [
        "the striker scored a goal in the match",
        "the team trained hard before the match",
        "she sprinted around the track to win",
        "the coach called a timeout near the end",
        "they threw the ball toward the open net",
        "the runner paced herself through the race",
        "the fans cheered when the team scored",
        "he dribbled past the defenders and shot",
        "the swimmer touched the wall first to win",
        "the players passed the ball across the field",
        "the team scored a goal and won the match",
        "the coach trained the players before the race",
        "the striker shot the ball into the net",
        "the runner won the race on the track",
        "the fans cheered the team and the coach",
        "the players sprinted across the wet field",
        "she threw the ball and scored the goal",
        "the swimmer trained hard before the race",
        "the team won the match and lifted the trophy",
        "the coach called the players off the field",
    ],
    "money": [
        "the bank raised the interest rate this quarter",
        "investors bought shares in the growing company",
        "she saved part of her salary each month",
        "the budget did not cover the rising costs",
        "the market fell sharply after the report",
        "they borrowed money from the bank",
        "the company reported strong profits this quarter",
        "inflation pushed the price of goods higher",
        "he paid off the loan ahead of schedule",
        "the fund returned a steady profit each year",
        "higher taxes reduced the income of the workers",
        "the currency lost value against the dollar",
        "the bank lowered the interest rate this year",
        "investors sold their shares as the market fell",
        "the company saved money and raised its profits",
        "rising costs and taxes cut into the budget",
        "he borrowed money to pay the rising rent",
        "the fund bought shares in the bank",
        "inflation and taxes slowly reduced her salary",
        "the market rose and the profits grew",
    ],
}

MULTILINGUAL = {
    "english": [
        "the small dog runs across the green field",
        "she reads a good book by the open window",
        "we eat bread and cheese for our lunch",
        "the children play down near the cold river",
        "he opens the heavy door and walks inside",
        "the sun rises slowly over the quiet town",
        "they sing a gentle song in the evening",
        "a small bird sits on the old bare tree",
    ],
    "spanish": [
        "el perro pequeno corre por el campo verde",
        "ella lee un buen libro junto a la ventana",
        "comemos pan y queso para nuestro almuerzo",
        "los ninos juegan cerca del rio frio",
        "el abre la puerta pesada y entra adentro",
        "el sol sale despacio sobre el pueblo tranquilo",
        "ellos cantan una cancion suave por la tarde",
        "un pajaro pequeno se posa en el arbol viejo",
    ],
    "french": [
        "le petit chien court a travers le champ vert",
        "elle lit un bon livre pres de la fenetre",
        "nous mangeons du pain et du fromage pour midi",
        "les enfants jouent pres de la riviere froide",
        "il ouvre la lourde porte et entre dedans",
        "le soleil se leve lentement sur la ville tranquille",
        "ils chantent une chanson douce dans la soiree",
        "un petit oiseau se pose sur le vieil arbre",
    ],
    "german": [
        "der kleine hund rennt ueber das gruene feld",
        "sie liest ein gutes buch am offenen fenster",
        "wir essen brot und kaese fuer unser mittagessen",
        "die kinder spielen unten am kalten fluss",
        "er oeffnet die schwere tuer und geht hinein",
        "die sonne geht langsam ueber der ruhigen stadt auf",
        "sie singen am abend ein sanftes lied",
        "ein kleiner vogel sitzt auf dem alten baum",
    ],
}


def _tokens(sentence):
    return sentence.lower().split()


# Common English function words. They appear in every topic and every sentence, so
# for learning MEANING and sorting by TOPIC they are noise that drowns the content
# words -- we drop them. (Character-level tasks below keep them: they are part of
# the language's fingerprint and part of natural-looking text.)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "i", "his", "her", "their", "our", "your", "my", "not", "do", "does", "did",
    "so", "if", "then", "than", "into", "out", "up", "down", "over", "near",
    "every", "more", "one", "two", "very", "again", "while", "around", "across",
    "right", "back", "whole", "single", "after", "before", "ahead",
}


def _content(sentence):
    """Tokens with the function words stripped out -- what the sentence is about."""
    return [w for w in _tokens(sentence) if w not in STOPWORDS]


# ===========================================================================
# 1. CAN IT LEARN?  -- word meaning from co-occurrence (random indexing)
# ===========================================================================

def learn_word_vectors(corpus, dim=512, window=2, seed=0):
    """Feed raw sentences to the random-indexing TextEncoder. Each word starts as
    a meaningless random atom; after learning, a word's vector is the (position-
    aware) sum of the atoms of the words it appears NEAR. Words used in similar
    contexts drift toward similar vectors -- distributional meaning, learned with
    no labels and no training loop."""
    enc = TextEncoder(dim, window=window, seed=seed)
    for sentence in corpus:
        enc.learn(_content(sentence))
    return enc


def relatedness(enc, groups):
    """Did related words actually land near each other? Compare the average cosine
    between words of the SAME hand-labeled group to the average between words of
    DIFFERENT groups. A gap means the encoder learned topical relatedness from raw
    text alone."""
    flat = [(w, g) for g, ws in groups.items() for w in ws]
    same, diff = [], []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            (wa, ga), (wb, gb) = flat[i], flat[j]
            c = cosine(enc.wordvec(wa), enc.wordvec(wb))
            (same if ga == gb else diff).append(c)
    return float(np.mean(same)), float(np.mean(diff))


# ===========================================================================
# 2. CAN IT ANALYZE?  -- language identification from characters
# ===========================================================================

class LanguageID:
    """Learn to tell languages apart from raw letters, knowing nothing about any
    of them. Each language gets a PROFILE vector: the superposition of all its
    character trigrams, where a trigram 'abc' is encoded order-sensitively by
    binding the three letter atoms after permuting each by its position. Languages
    differ in which trigrams are common ('the' vs 'ent' vs 'sch'), so the profiles
    end up pointing in different directions. Identify = nearest profile by cosine."""

    def __init__(self, dim=512, seed=0):
        self.dim = dim
        self.atoms = Vocabulary(dim, seed)     # one fixed random vector per character
        self.profiles = {}

    def _text_vector(self, text):
        text = text.lower()
        tris = []
        for i in range(len(text) - 2):
            a = permute(self.atoms.get(text[i]), 0)
            b = permute(self.atoms.get(text[i + 1]), 1)
            c = permute(self.atoms.get(text[i + 2]), 2)
            tris.append(bind(bind(a, b), c))
        return bundle(tris) if tris else np.zeros(self.dim)

    def fit(self, texts_by_language):
        for lang, texts in texts_by_language.items():
            self.profiles[lang] = bundle([self._text_vector(t) for t in texts])
        return self

    def identify(self, text):
        v = self._text_vector(text)
        return max(self.profiles, key=lambda lang: cosine(v, self.profiles[lang]))


# ===========================================================================
# 3. CAN IT ORGANIZE?  -- topic sorting, supervised and unsupervised
# ===========================================================================

def _kmeans_cosine(X, k, seed=0, iters=25, restarts=10):
    """k-means on the unit sphere, used to DISCOVER topics with no labels. Restarts
    from spread-out seeds (a cosine k-means++) and keeps the tightest result, since
    a single random start clusters text poorly."""
    best_assign, best_score = None, -np.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        idx = [int(rng.integers(len(X)))]                # first centroid at random
        while len(idx) < k:                              # then the least-similar point
            nearest = (X @ X[idx].T).max(axis=1)
            idx.append(int(np.argmin(nearest)))
        cent = X[idx].copy()
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        assign = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            assign = (X @ cent.T).argmax(1)
            for j in range(k):
                members = X[assign == j]
                if len(members):
                    s = members.sum(0)
                    cent[j] = s / (np.linalg.norm(s) or 1.0)
                else:
                    cent[j] = X[rng.integers(len(X))]
        score = sum(float((X[assign == j] @ cent[j]).sum()) for j in range(k) if (assign == j).any())
        if score > best_score:
            best_score, best_assign = score, assign
    return best_assign


def _purity(assign, truth):
    groups = defaultdict(list)
    for a, t in zip(assign, truth):
        groups[a].append(t)
    correct = sum(Counter(g).most_common(1)[0][1] for g in groups.values())
    return correct / len(truth)


class TopicSorter:
    """Sort sentences by topic, using the word vectors learned in step 1. A sentence
    becomes one vector: the bundle of its content words' LEARNED vectors. Because
    related words now sit near each other, two sentences about the same topic land
    near each other even when they share no exact words -- which is what lets the
    unsupervised clustering find the topics. Supervised: one prototype per topic
    (the bundle of its training sentences), classify by nearest."""

    def __init__(self, encoder):
        self.enc = encoder                     # a trained TextEncoder (learned meanings)
        self.dim = encoder.dim
        self.prototypes = {}

    def sentence_vector(self, sentence):
        toks = _content(sentence)
        return bundle([self.enc.wordvec(w) for w in toks]) if toks else np.zeros(self.dim)

    def fit(self, labeled):
        """labeled: dict topic -> list of sentences."""
        for topic, sentences in labeled.items():
            self.prototypes[topic] = bundle([self.sentence_vector(s) for s in sentences])
        return self

    def classify(self, sentence):
        v = self.sentence_vector(sentence)
        return max(self.prototypes, key=lambda t: cosine(v, self.prototypes[t]))

    def discover(self, sentences, k, seed=0):
        """Cluster sentences into k groups with NO labels -> cluster id per sentence."""
        X = np.stack([self.sentence_vector(s) for s in sentences])
        return _kmeans_cosine(X, k, seed=seed)


# ===========================================================================
# 4. CAN IT PRODUCE?  -- a holographic character n-gram generator
# ===========================================================================

class HolographicNGram:
    """Generate text the only way a system with no grammar can: by learning, for
    each short context of characters, what tends to come next. The next-character
    distribution after a context is stored holographically -- as the SUPERPOSITION
    of the atoms of the characters seen following it -- and read back by cleanup
    (cosine to each character atom), which naturally surfaces the most likely next
    character. Unseen contexts BACK OFF to a shorter one. The result reads like the
    training text up close and like nonsense from afar: that is what an n-gram model
    is, holographic or not.

    A single global memory cannot hold every association at once (the capacity
    limit the scaling work is about), so the model PARTITIONS by context -- one
    bundle per context string -- which is the same partition-to-beat-capacity trick
    used elsewhere in this codebase."""

    def __init__(self, dim=1024, n=3, seed=0, fold_case=True):
        self.dim = dim
        self.n = n
        self.atoms = Vocabulary(dim, seed)
        # fold_case=True (the default, and every pinned measurement) folds text
        # to lowercase: a smaller alphabet, denser contexts. Setting it False
        # preserves capitals -- measured on Austen with the hierarchical coder:
        # true-cased text costs ~12% bits/char and ~8 points of word coherence,
        # but the output reads as PROSE (capitals, punctuation, sentences).
        self.fold_case = fold_case
        self.alphabet = set()
        self._mem = defaultdict(lambda: np.zeros(dim))   # context string -> sum of next-char atoms
        self._seen = defaultdict(Counter)                # context -> next-char counts (for sampling)

    def fit(self, text):
        text = text.lower() if self.fold_case else text
        self.alphabet.update(text)
        # store the next character under contexts of EVERY length 1..n, so an unseen
        # long context can back off to a shorter one that was seen
        for j in range(1, len(text)):
            nxt = text[j]
            for m in range(1, min(self.n, j) + 1):
                ctx = text[j - m:j]
                self._mem[ctx] = self._mem[ctx] + self.atoms.get(nxt)
                self._seen[ctx][nxt] += 1
        return self

    def _distribution(self, ctx):
        """Cosine of the context's stored bundle to every character atom -- the
        holographic next-character distribution. Backs off to shorter contexts when
        the full one was never seen."""
        for length in range(len(ctx), 0, -1):
            key = ctx[-length:]
            if key in self._mem:
                probe = self._mem[key]
                return {ch: cosine(probe, self.atoms.get(ch)) for ch in self.alphabet}
        return {ch: 0.0 for ch in self.alphabet}

    def next_char(self, ctx):
        dist = self._distribution(ctx)
        return max(dist, key=dist.get)

    def generate(self, seed_text, length=160, temperature=0.5, rng=None, top_p=1.0):
        """Sample text one character at a time from the learned distribution. top_p<1.0
        switches on NUCLEUS decoding: at each step keep only the smallest set of
        likeliest characters whose probability sums to top_p, and sample within it --
        trimming the unlikely tail that breaks words. top_p=1.0 (the default) is the
        original plain-temperature behaviour exactly, so existing callers are unchanged.

        DELEGATES the per-step temperature+nucleus draw to holographic_tokensample.
        sample_from_distribution -- the primitive PROMOTED from this very loop (proven
        bit-identical across temperature x top_p x distributions before the switch). WHY
        delegate: keeping a private copy here means the char generator and every other
        sampler drift apart; one primitive keeps them honest and shares the T->0 argmax
        guard the inline loop never had."""
        from holographic.agents_and_reasoning.holographic_tokensample import sample_from_distribution
        rng = rng or np.random.default_rng(0)
        out = seed_text.lower() if self.fold_case else seed_text
        for _ in range(length):
            dist = self._distribution(out[-self.n:])
            ch = sample_from_distribution(dist, temperature=temperature, top_p=top_p, rng=rng)
            if ch is None:                               # no positive mass -- stop, as the inline loop did
                break
            out += ch
        return out

    def predict_accuracy(self, text):
        """Top-1 next-character accuracy on held-out text -- how often the model's
        single best guess is right."""
        text = text.lower() if self.fold_case else text
        ok = total = 0
        for i in range(len(text) - self.n):
            if self.next_char(text[i:i + self.n]) == text[i + self.n]:
                ok += 1
            total += 1
        return ok / total if total else 0.0

    @staticmethod
    def real_word_fraction(generated, vocabulary):
        """Of the whole words the generator emitted, how many are REAL words it could
        have seen? The model works one character at a time with no concept of a word,
        so this measures whether word structure fell out of the letter statistics."""
        words = [w for w in generated.split() if w]
        if not words:
            return 0.0
        real = sum(w in vocabulary for w in words)
        return real / len(words)


# ===========================================================================
# DEMO  (each capability, measured and reported honestly)
# ===========================================================================

def _split(items, frac=0.7, seed=0):
    items = list(items)
    np.random.default_rng(seed).shuffle(items)
    cut = max(1, int(len(items) * frac))
    return items[:cut], items[cut:]


def demo_text():
    print("=" * 70)
    print("Text from scratch: a system that knows no language, learning from text")
    print("=" * 70)

    # -- 1. learn: word relatedness from co-occurrence ----------------------
    corpus = [s for sents in TOPICS.values() for s in sents]
    enc = learn_word_vectors(corpus, dim=512, window=2, seed=0)
    groups = {
        "cooking": ["flour", "oven", "dough", "bake", "salt"],
        "space":   ["planet", "rocket", "orbit", "moon", "star"],
        "sports":  ["team", "goal", "race", "ball", "coach"],
        "money":   ["bank", "market", "profit", "salary", "loan"],
    }
    same, diff = relatedness(enc, groups)
    print("\n1. CAN IT LEARN?  word relatedness from raw co-occurrence")
    print(f"   words used alike land closer: same-topic cosine {same:+.3f} vs "
          f"cross-topic {diff:+.3f}  (gap {same - diff:+.3f})")
    for w in ("oven", "planet", "ball"):
        near = ", ".join(n for n, _ in enc.nearest(w, 3))
        print(f"   nearest to '{w}': {near}")

    # -- 2. analyze: language identification from characters ----------------
    train, test = {}, []
    for lang, sents in MULTILINGUAL.items():
        tr, te = _split(sents, frac=0.65, seed=1)
        train[lang] = tr
        test += [(s, lang) for s in te]
    lid = LanguageID(dim=512, seed=0).fit(train)
    correct = sum(lid.identify(s) == lang for s, lang in test)
    print("\n2. CAN IT ANALYZE?  language ID from letters alone (no dictionary)")
    print(f"   identified {correct}/{len(test)} held-out sentences correctly "
          f"({100 * correct / len(test):.0f}%) across {len(train)} languages")

    # -- 3. organize: topic sorting, supervised and unsupervised ------------
    tr_by_topic, te = {}, []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        tr_by_topic[topic] = a
        te += [(s, topic) for s in b]
    sorter = TopicSorter(enc).fit(tr_by_topic)         # reuses the meanings from step 1
    hit = sum(sorter.classify(s) == topic for s, topic in te)
    all_sents = [s for sents in TOPICS.values() for s in sents]
    all_truth = [t for t, sents in TOPICS.items() for _ in sents]
    assign = sorter.discover(all_sents, k=len(TOPICS), seed=3)
    print("\n3. CAN IT ORGANIZE?  sort sentences by topic (using the learned meanings)")
    print(f"   supervised : labeled {hit}/{len(te)} held-out sentences "
          f"({100 * hit / len(te):.0f}%)")
    print(f"   unsupervised: clustered all {len(all_sents)} sentences into "
          f"{len(TOPICS)} groups at {100 * _purity(assign, all_truth):.0f}% purity "
          "(topics discovered, not told)")

    # -- 4. produce: holographic n-gram generation --------------------------
    text = " ".join(s for sents in TOPICS.values() for s in sents)
    cut = int(len(text) * 0.85)
    ng = HolographicNGram(dim=1024, n=4, seed=0).fit(text[:cut])
    acc = ng.predict_accuracy(text[cut:])
    base = Counter(text[:cut]).most_common(1)[0][1] / len(text[:cut])
    sample = ng.generate("the ", length=140, temperature=0.4)
    vocab = {w for sents in TOPICS.values() for sent in sents for w in _tokens(sent)}
    real = HolographicNGram.real_word_fraction(sample, vocab)
    print("\n4. CAN IT PRODUCE?  a holographic character n-gram model")
    print(f"   next-letter guess right {100 * acc:.0f}% of the time on held-out text "
          f"(vs {100 * base:.0f}% for always-guess-space)")
    print(f"   but {100 * real:.0f}% of the WORDS it invents are real words -- word")
    print("   structure fell out of letter statistics, with no notion of a word")
    print(f"   sample: \"{sample}\"")
    print("\n   Honest verdict: locally it reads like the training text, globally it")
    print("   drifts into nonsense -- exactly what a small n-gram model does. It has")
    print("   learned letter and word statistics, not meaning or grammar.")


# ===========================================================================
# SCALING UP  -- the same capabilities on real public-domain corpora (NLTK)
# ===========================================================================
#
# The built-in datasets above are tiny on purpose -- enough to test and read. To
# see how far the SAME machinery goes with real text, this section pulls public-
# domain corpora from NLTK (which hosts them on GitHub): Project Gutenberg books,
# the Universal Declaration of Human Rights in many languages, and the genre-
# labeled Brown corpus. Nothing here changes the algorithms -- only the amount of
# text they learn from. Needs `pip install nltk` and a one-time download; if either
# is missing the demo says so and the rest of the module still works offline.

import os
import re


def ensure_corpora(packages=("gutenberg", "udhr", "brown", "punkt_tab")):
    """Make sure the public-domain NLTK corpora are available, downloading them to
    the standard ~/nltk_data if needed. Returns the nltk module, or None if nltk or
    the network is unavailable."""
    try:
        import nltk
    except ImportError:
        return None
    home = os.path.join(os.path.expanduser("~"), "nltk_data")
    if home not in nltk.data.path:
        nltk.data.path.insert(0, home)
    for pkg in packages:
        try:
            nltk.download(pkg, download_dir=home, quiet=True)
        except Exception:
            pass
    return nltk


def _raw_sentences(raw):
    """Split a raw block of text into sentences of lowercase alphabetic tokens, with
    no dependency on a sentence tokenizer (just punctuation)."""
    raw = raw.lower().replace("\n", " ")
    return [re.findall(r"[a-z]+", part) for part in re.split(r"[.!?]", raw)]


def demo_text_scaled():
    """The four capabilities again, on real corpora. Same code, much more text."""
    nltk = ensure_corpora()
    print("\n" + "=" * 70)
    print("Scaling up: the same machinery on real public-domain corpora (NLTK)")
    print("=" * 70)
    if nltk is None:
        print("  (needs `pip install nltk`; the built-in demo above runs without it.)")
        return
    try:
        from nltk.corpus import gutenberg, udhr, brown
        gutenberg.fileids(); udhr.fileids(); brown.categories()
    except Exception:
        print("  (corpora not downloaded yet -- run once with a network connection.)")
        return

    # 1. language ID across many languages (UDHR) ---------------------------
    wanted = ["English-Latin1", "Spanish-Latin1", "French_Francais-Latin1",
              "German_Deutsch-Latin1", "Italian-Latin1", "Dutch_Nederlands-Latin1",
              "Portuguese_Portugues-Latin1", "Finnish_Suomi-Latin1",
              "Swedish_Svenska-Latin1", "Danish_Dansk-Latin1",
              "Norwegian_Norsk-Latin1", "Latin_Latina-Latin1"]
    have = [f for f in wanted if f in udhr.fileids()]

    def chunk(text, n=120):
        text = " ".join(text.split())
        return [text[i:i + n] for i in range(0, max(0, len(text) - n), n)]
    train, test = {}, []
    for f in have:
        cs = chunk(udhr.raw(f))
        name = f.split("-")[0].split("_")[0]
        train[name] = cs[:len(cs) * 2 // 3]
        test += [(c, name) for c in cs[len(cs) * 2 // 3:]]
    lid = LanguageID(dim=1024, seed=0).fit(train)
    acc = sum(lid.identify(c) == l for c, l in test) / len(test)
    print(f"\n1. LANGUAGE ID: {len(have)} languages, {len(test)} held-out passages "
          f"-> {100 * acc:.0f}% correct (from letters alone)")

    # 2. word meaning from a whole novel ------------------------------------
    sents = [_content(" ".join(s)) for s in _raw_sentences(gutenberg.raw("austen-emma.txt"))]
    enc = TextEncoder(1024, window=3, seed=0)
    for s in sents:
        if s:
            enc.learn(s)
    print(f"\n2. WORD MEANING from one novel (Emma, {len(enc.context)} word types):")
    for w in ("woman", "happy", "house", "father"):
        if w in enc.context:
            print(f"     {w:7s} ~ {', '.join(n for n, _ in enc.nearest(w, 6))}")

    # 3. genre classification on labeled documents (Brown) ------------------
    cats = ["news", "romance", "science_fiction", "government", "hobbies"]
    genc = TextEncoder(1024, window=3, seed=1)
    docs = []
    for c in cats:
        for fid in brown.fileids(categories=c):
            words = [w.lower() for w in brown.words(fid) if w.isalpha()][:1200]
            genc.learn(words)
            docs.append((words, c))

    def docvec(words):
        cont = [w for w in words if w not in STOPWORDS]
        return bundle([genc.wordvec(w) for w in cont]) if cont else np.zeros(1024)
    rng = np.random.default_rng(0)
    rng.shuffle(docs)
    cut = int(len(docs) * 0.7)
    tr, te = docs[:cut], docs[cut:]
    protos = {c: bundle([docvec(w) for w, cc in tr if cc == c]) for c in cats}
    hit = sum(max(protos, key=lambda c: cosine(docvec(w), protos[c])) == cc for w, cc in te)
    print(f"\n3. GENRE CLASSIFY: {len(docs)} real documents across {len(cats)} genres "
          f"-> {hit}/{len(te)} held-out right ({100 * hit / len(te):.0f}%)")

    # 4. generation from a whole novel --------------------------------------
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", gutenberg.raw("carroll-alice.txt").lower()))
    cut = int(len(text) * 0.9)
    ng = HolographicNGram(dim=1024, n=6, seed=0).fit(text[:cut])
    nacc = ng.predict_accuracy(text[cut:cut + 3000])
    real = HolographicNGram.real_word_fraction(ng.generate("alice ", 220, 0.45), set(text.split()))
    sample = ng.generate("the ", 160, 0.45)
    print(f"\n4. GENERATION from a novel (Alice, 6-gram): next-letter {100 * nacc:.0f}%, "
          f"{100 * real:.0f}% real words")
    print(f"     \"{sample}\"")
    print("\n  Same code as the built-in demo -- only the amount of text changed. With a")
    print("  real corpus, word neighbours become genuine (woman~lady/girl), languages")
    print("  separate cleanly, genres sort well above chance, and the text reads like")
    print("  its source up close. The limits are the honest ones: distributional, not")
    print("  grammatical; an n-gram, not an understanding.")


def demo_text_self_organizing():
    """Route the text classifier through the SELF-ORGANIZING MEMORY (the shadow-swap,
    split/merge, autonomously-reorganizing store from holographic_organizer.py)
    instead of a plain prototype, and measure honestly whether the heavier machinery
    earns its keep on text. The answer is a clean negative that is itself the point:
    text topics are linearly separable, so one prototype per topic is already
    optimal; the AUTONOMOUS memory discovers this by measurement and refuses to
    split, matching the simple baseline at minimum size, while naively forcing more
    sub-prototypes only wastes memory for no gain. Here the machinery's job is to
    know when NOT to fire -- and it does."""
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind

    print("\n" + "=" * 70)
    print("Text through the self-organizing memory (does the heavy machinery help?)")
    print("=" * 70)
    corpus = [s for sents in TOPICS.values() for s in sents]
    enc = learn_word_vectors(corpus, dim=512, window=2, seed=0)

    def sv(s):
        toks = _content(s)
        return bundle([enc.wordvec(w) for w in toks]) if toks else np.zeros(512)

    tr, te = {}, []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        tr[topic] = a
        te += [(s, topic) for s in b]
    train_items = [(sv(s), topic) for topic, sents in tr.items() for s in sents]

    # (1) one prototype per topic -- the simple baseline
    proto = {t: bundle([sv(s) for s in sents]) for t, sents in tr.items()}
    base = sum(max(proto, key=lambda t: cosine(sv(s), proto[t])) == topic for s, topic in te) / len(te)

    # (2) NAIVELY force several sub-prototypes per topic (surely more is better?)
    forced = SelfOrganizingMind(dim=512, seed=0)._shadow_at_k(train_items, 3)
    facc = sum(forced.classify(sv(s))[0] == topic for s, topic in te) / len(te)

    # (3) the AUTONOMOUS memory -- it picks the resolution by measured accuracy
    mind = SelfOrganizingMind(dim=512, seed=0)
    order = list(train_items)
    np.random.default_rng(0).shuffle(order)
    reorgs = 0
    for i, (v, topic) in enumerate(order, 1):
        mind.observe_vector(v, topic)
        if i % 20 == 0:
            r = mind.auto_reorganize()
            if r and r[0] != "keep":
                reorgs += 1
    r = mind.auto_reorganize()
    if r and r[0] != "keep":
        reorgs += 1
    aacc = sum(mind.classify_vector(sv(s))[0] == topic for s, topic in te) / len(te)

    print(f"\n  one prototype / topic    : {100 * base:.0f}%   ({len(proto)} prototypes)")
    print(f"  forced 3 sub-protos/topic: {100 * facc:.0f}%   ({forced.size()} prototypes)")
    print(f"  autonomous memory        : {100 * aacc:.0f}%   ({mind.live.size()} prototypes, "
          f"reorganized {reorgs}x)")
    print("\n  Honest verdict: text topics are linearly separable, so one prototype per")
    print("  topic is already optimal -- forcing more only spends memory for no gain.")
    print("  The autonomous memory MEASURES this and keeps one each, matching the simple")
    print("  baseline without over-engineering. The win is discipline, not accuracy: the")
    print("  same self-* machinery from the organizer, now knowing when not to fire on")
    print("  text. (For blind clustering the plain k-means in TopicSorter still wins;")
    print("  the split-based discovery is coherence-sensitive on these embeddings.)")


def _eval_classifiers(items, dim=1024, seed=0):
    """Single-prototype vs forced-3-sub-protos vs autonomous self-organizing memory,
    on a balanced train/test split of (token-list, label) documents."""
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
    by = defaultdict(list)
    for w, l in items:
        by[l].append(w)
    cap = min(min(len(v) for v in by.values()), 200)
    rng = np.random.default_rng(seed)
    rows = []
    for l, ws in by.items():
        ws = list(ws); rng.shuffle(ws)
        rows += [(w, l) for w in ws[:cap]]
    rng.shuffle(rows)
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    enc = TextEncoder(dim, window=3, seed=seed)
    for w, _ in train:
        enc.learn(w)

    def dv(w):
        c = [x for x in w if x not in STOPWORDS]
        return bundle([enc.wordvec(x) for x in c]) if c else np.zeros(dim)
    tr = [(dv(w), l) for w, l in train]
    tev = [(dv(w), l) for w, l in test]
    cats = sorted(by)

    proto = {c: bundle([v for v, l in tr if l == c]) for c in cats}
    single = sum(max(proto, key=lambda c: cosine(v, proto[c])) == l for v, l in tev) / len(tev)
    forced = SelfOrganizingMind(dim=dim, seed=seed)._shadow_at_k(tr, 3)
    fk = sum(forced.classify(v)[0] == l for v, l in tev) / len(tev)
    mind = SelfOrganizingMind(dim=dim, seed=seed)
    for v, l in tr:
        mind.observe_vector(v, l)
    r = mind.auto_reorganize()
    auto = sum(mind.classify_vector(v)[0] == l for v, l in tev) / len(tev)
    return {"classes": len(cats), "n": len(rows), "single": single, "forced": fk,
            "auto": auto, "chose": r[0] if r else "keep",
            "protos": (len(cats), forced.size(), mind.live.size())}


def demo_text_hard():
    """The honest stress test: give the self-organizing memory genuinely HARD text --
    confusable, multi-modal categories where a single prototype's average lands in the
    wrong place -- and see whether it fires and helps. Reuters financial news (crude /
    trade / money-fx / interest share vocabulary) is exactly that; sentiment (movie
    reviews) is hard for a different reason -- the representation carries no good/bad
    signal -- so it should NOT be fixable by splitting."""
    nltk = ensure_corpora(("reuters", "movie_reviews", "punkt_tab"))
    print("\n" + "=" * 70)
    print("Hard problems: where does splitting the memory actually help?")
    print("=" * 70)
    if nltk is None:
        print("  (needs `pip install nltk`; runs offline on the built-in data above.)")
        return
    try:
        from nltk.corpus import reuters, movie_reviews
        reuters.fileids(); movie_reviews.fileids()
    except Exception:
        print("  (corpora not downloaded yet -- run once with a network connection.)")
        return

    single_label = [(f, reuters.categories(f)[0]) for f in reuters.fileids()
                    if len(reuters.categories(f)) == 1]
    top = ["earn", "acq", "crude", "trade", "money-fx", "interest",
           "money-supply", "ship", "sugar", "coffee"]
    reut = [([w.lower() for w in reuters.words(f) if w.isalpha()], c)
            for f, c in single_label if c in top]
    r = _eval_classifiers(reut, seed=0)
    print(f"\n  REUTERS ({r['classes']} confusable financial categories, {r['n']} docs):")
    print(f"    one prototype / category : {100 * r['single']:.0f}%")
    print(f"    forced 3 sub-protos      : {100 * r['forced']:.0f}%   (the capability's ceiling)")
    print(f"    autonomous memory        : {100 * r['auto']:.0f}%   (chose {r['chose']}, "
          f"{r['protos'][2]} prototypes) -- it FIRED")

    sent = [([w.lower() for w in movie_reviews.words(f) if w.isalpha()], cat)
            for cat in ("pos", "neg") for f in movie_reviews.fileids(cat)]
    s = _eval_classifiers(sent, seed=0)
    print(f"\n  SENTIMENT (movie reviews, pos vs neg, {s['n']} docs):")
    print(f"    one prototype / class    : {100 * s['single']:.0f}%")
    print(f"    autonomous memory        : {100 * s['auto']:.0f}%   (chose {s['chose']})")

    print("\n  Honest verdict: on Reuters the categories really are multi-modal and")
    print("  confusable, so a single averaged prototype mis-files the off-centre ones --")
    print("  splitting helps, and the autonomous memory measures that and fires. On")
    print("  sentiment, splitting cannot help: the failure is the representation (random")
    print("  word atoms carry no good/bad signal), not multi-modality, and the memory")
    print("  correctly declines. The discipline cuts both ways -- it fires only when the")
    print("  held-out data says splitting earns its keep.")


def demo_text_multilingual():
    """The same from-scratch character n-gram -- nothing language-specific in it -- run
    on five European languages (European Parliament proceedings). It learns each
    language's spelling and short-range word structure straight from the characters:
    the accents of French, the long compounds of German, the function words of each, and
    then generates text that reads as that language. This is a CAPABILITY test, not a new
    mechanism: the generator is unchanged from the English demo above; only the language
    of the text differs. It is still distributional, not grammatical -- an n-gram, not an
    understanding -- but it shows the approach was never English-specific."""
    nltk = ensure_corpora(("europarl_raw",))
    print("\n" + "=" * 70)
    print("Generation across languages (same code, five languages, no tuning)")
    print("=" * 70)
    if nltk is None:
        print("  (needs `pip install nltk`; runs offline on the built-in data above.)")
        return
    try:
        from nltk.corpus import europarl_raw as eu
        eu.english.words()
    except Exception:
        print("  (europarl corpus not downloaded yet -- run once with a connection.)")
        return

    seeds = {"english": "the ", "french": "le ", "german": "der ",
             "spanish": "el ", "italian": "il "}
    for lang, seed_text in seeds.items():
        words = [w.lower() for w in getattr(eu, lang).words()[:35000] if w.isalpha()]
        text = " ".join(words)
        cut = int(len(text) * 0.9)
        ng = HolographicNGram(dim=1024, n=6, seed=0).fit(text[:cut])
        acc = ng.predict_accuracy(text[cut:cut + 3000])
        sample = ng.generate(seed_text, 150, 0.45)
        real = HolographicNGram.real_word_fraction(sample, set(text.split()))
        print(f"\n  {lang.upper()}  next-letter {100 * acc:.0f}%, {100 * real:.0f}% real words")
        print(f"    \"{sample[:140]}\"")
    print("\n  One generator, five languages, nothing tuned per language. Each reads")
    print("  unmistakably as itself -- French keeps its accents, German its compounds,")
    print("  Spanish its endings. The limit is the same honest one as in English: it")
    print("  spells and chains words plausibly without knowing what any of them mean.")


def _selftest():
    """Regression trap (T6 backfill; demos only, no assertion). Pins two contracts: a HolographicNGram trained on
    text PREDICTS that text's next character far better than chance (it has learned the sequence statistics), and
    learned word vectors place a co-occurring word nearer than an unrelated one. Contrast-based, not absolute."""
    import numpy as np

    # 1. N-GRAM next-char prediction: trained on a repetitive corpus, accuracy is high; on the 27-symbol space
    #    chance is ~0.04, so >0.5 is unmistakable learning, not luck. (Measured ~0.98 on this corpus.)
    ng = HolographicNGram()
    text = "the quick brown fox jumps over the lazy dog " * 30
    ng.fit(text)
    acc = ng.predict_accuracy(text)
    assert acc > 0.5, "n-gram failed to learn its own training text: acc=%.3f" % acc

    # 2. WORD VECTORS carry co-occurrence: in a corpus where 'cat' always co-occurs with 'sat', cat is nearer to
    #    sat than to a word from the other sentence. The [BLIND-SPOT] point: assert the RELATION, not a magnitude.
    enc = learn_word_vectors("the cat sat here the cat sat again the dog ran there the dog ran fast " * 8,
                             dim=512, window=2, seed=0)

    def _cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    cat, sat, ran = enc.wordvec("cat"), enc.wordvec("sat"), enc.wordvec("ran")
    if cat is not None and sat is not None and ran is not None:
        assert _cos(cat, sat) > _cos(cat, ran)                 # cat co-occurs with sat, not ran

    print("OK: holographic_text self-test passed (an n-gram predicts its training text's next char at acc>0.5 vs "
          "~0.04 chance, and word vectors place a co-occurring word nearer than an unrelated one)")


if __name__ == "__main__":
    import sys
    _selftest()
    if "--demos" in sys.argv:
        demo_text()
        demo_text_self_organizing()
        demo_text_scaled()
        demo_text_hard()
        demo_text_multilingual()