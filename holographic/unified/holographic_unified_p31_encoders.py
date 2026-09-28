"""UnifiedMind part 31 -- encoders as plugins of the substrate (the CLM backlog, phase F).

E6.1  superposed frozen encoders: several frozen text channels (character n-grams, word hashing, a WordNet synonym
      channel from the vendored dictionary) bound under unitary role keys and summed into one hypervector, with one
      learned relevance per channel (GRLVQ-style). The machinery is holographic_superposed; the measurement is
      tools/bench_superposed.py -> docs/research/evidence/bench_superposed.json.
E6.2 / E6.3  the model-end tier and its state encoder are NOT here: they need a model outside the wheel, so they
      are the bundled `clm` plugin (holographic/plugins/clm.py; docs/PLUGINS.md). Its encoder reaches a mind
      through the existing set_embedder seam, whose own verify gate decides.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by holographic/misc/
holographic_unified.py (the only import path anyone uses). Carries no `__init__`; per-mind state lives in
self.__dict__ under underscore names, created on first use.
"""
from holographic.unified import check_part


class _UnifiedPart31:

    def superposed_encoder(self, channels=None, dim=2048, seed=0):
        """Build a text encoder that superposes frozen channels (char n-grams, word hashing, WordNet synonyms)
        under unitary role keys; returns a SuperposedEncoder whose .encode(text) is one unit hypervector.

        channels: names from ("ngram", "word", "synonym") and/or (name, fn) pairs for a custom frozen channel;
        None = all three. The encoder is cached per (channels, dim, seed) on this mind. A channel that cannot be
        built (the synonym channel when the vendored dictionary is missing) is listed in .skipped with the reason
        rather than failing. .bound(text) gives the per-channel bound rows, .unbind(vec, name) reads one channel
        back out, .channel_scores(text, P) gives the per-channel scores a relevance learner needs.

        KEPT NEGATIVE, MEASURED (tools/bench_superposed.py, dim 2048, ProtoStore rule, one pass, 3 seeds, paired
        bootstrap): the superposition does NOT beat its best channel. The n-gram channel alone won on both
        datasets -- CLINC150 top-1 0.9029 vs superposed 0.8999 (-0.30, CI [-0.76, +0.19]); Banking77 0.8813 vs
        0.8761 (-0.52, CI [-1.14, +0.09]); AURC ties. Exact score fusion of the three channels (3x memory) was
        +0.43 [+0.01, +0.95] on CLINC, so the channels are complementary but one 2048-d vector's cross-talk eats
        it. For a text door, use the n-gram channel alone (hashed_ngram_encode); this encoder is for composing
        channels you want to unbind, not for accuracy. docs/research/evidence/bench_superposed.json has every
        number. See holographic_superposed."""
        from holographic.agents_and_reasoning.holographic_superposed import CHANNELS, SuperposedEncoder
        chans = tuple(CHANNELS if channels is None else channels)
        cache = self.__dict__.setdefault("_superposed_encoders", {})
        # custom channels are callables: key them by name and identity so two different callables never share
        key = (tuple((c[0], id(c[1])) if isinstance(c, (tuple, list)) else c for c in chans), int(dim), int(seed))
        enc = cache.get(key)
        if enc is None:
            enc = cache[key] = SuperposedEncoder(channels=chans, dim=dim, seed=seed)
        return enc

    def superposed_relevance(self, door="superposed", channels=None, lr=0.001):
        """Get the named door's per-channel relevance learner (GRLVQ-style, one scalar per channel, summing
        to 1); returns a ChannelRelevance -- feed it verdicts with .observe(scores, truth) and read .weights.

        One instance per door, created on first use (the panel's Q5 rule: one class, separate instances). The
        whole learned state is C floats (.state() / ChannelRelevance.from_state). lr is the GRLVQ step on the
        relevances; 0.001 is what the bench's validation split chose on BOTH datasets from {0.001, 0.01, 0.05}
        (CLINC150 val 0.8887 / 0.8857 / 0.8857; Banking77 0.8671 / 0.8585 / 0.8595 -- larger steps zero the word
        channel and score worse). KEPT NEGATIVE: with the chosen step the relevances recovered +0.22 top-1 of the
        superposition's loss on CLINC150 and lost 0.15 more on Banking77 (-0.67 vs n-grams alone, CI [-1.29,
        -0.08]). See holographic_superposed.ChannelRelevance."""
        from holographic.agents_and_reasoning.holographic_superposed import CHANNELS, ChannelRelevance
        rels = self.__dict__.setdefault("_superposed_relevances", {})
        r = rels.get(door)
        if r is None:
            r = rels[door] = ChannelRelevance(list(CHANNELS if channels is None else channels), lr=lr)
        elif channels is not None and list(channels) != r.channels:
            raise ValueError("door %r already learns relevances for %s, not %s" % (door, r.channels, list(channels)))
        return r

    def superposed_door(self, door="superposed", channels=None, dim=2048, seed=0, use_relevance=True):
        """Get a nearest-prototype door over the superposed encoder that learns from labelled verdicts; returns a
        SuperposedDoor with add_option(label, texts), rank(text) and learn(text, truth).

        Its prototypes are this mind's protostore(door) (the one shared InfoNCE rule, fed the UNIFORM
        superposition) and its relevances are superposed_relevance(door) -- so learned state lives in the same
        per-door stores every other door uses. use_relevance=False ranks the uniform superposition (the bench's
        'superposed' arm); True ranks the relevance-weighted query (the 'superposed_rel' arm). The two share one
        prototype trajectory. See holographic_superposed.SuperposedDoor."""
        from holographic.agents_and_reasoning.holographic_superposed import SuperposedDoor
        enc = self.superposed_encoder(channels=channels, dim=dim, seed=seed)
        store = self.protostore(door, dim=dim)
        rel = self.superposed_relevance(door, channels=enc.channels)
        return SuperposedDoor(enc, store=store, relevance=rel, use_relevance=use_relevance, name=door)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p31_encoders", "_UnifiedPart31")
    print("holographic_unified_p31_encoders selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
