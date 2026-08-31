# leCore with an LLM attached: the full-advantage quickstart

    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(my_llm)              # my_llm: prompt(str) -> str. That's the whole wiring.

    m.ask("question")                 # the ANSWER LADDER: reflex -> your knowledge -> tools
                                      # -> (intern) -> the model LAST. Provenance on every
                                      # answer; cheap rungs refuse rather than guess.
    m.do("multi-step request")        # ONE plan call (zero on repeats via learned skeletons);
                                      # tool steps run without the model; chains are learned.
    m.zoo_idle()                      # answer questions nobody asked yet: void map -> prefill
                                      # of the FREE rungs only.
    m.zoo_report()                    # the ledger: per-tier serves, est tokens saved,
                                      # mined skeletons, learned transitions.

Measured on the built-in 20-query mixed workload (asserted in the Part-20 selftest): the naive
integration spends 20 model calls; the attached mind spends 2 (10x fewer), with the one
genuinely unanswerable question ESCALATING to the model instead of being served a wrong cached
answer. Synthesis (synthesize_tool / synthesize_tool_certified, with Lean 4 well-typedness
certificates) registers tools that ask()/do() pick up automatically.
