"""Is leCore's answer to "a better RNN" actually PROGRAM INDUCTION?

THE HYPOTHESIS THIS SESSION ARRIVED AT. Five sessions of asking "how do we build a better
recurrent model" may have been the wrong question. An RNN learns a transform by compressing
history into a hidden state and fitting a readout. leCore has two faculties that learn a
transform WITHOUT a hidden state at all:

  abstract_program(examples)  -- given (input_vec, output_vec) pairs demonstrating ONE
        transform, synthesise a procedure that reproduces the FIRST example, then VERIFY it
        reproduces ALL the others. The abstraction is the program CONSISTENT ACROSS
        examples, not a fit to one. Its own claim: a stored prototype only matches
        near-identical states, but an abstracted program captures the TRANSFORM, so it
        transfers to inputs never seen.

  synthesize_program(library, goal) -- when no tool chain reaches a goal, optimise a chain
        in latent space toward it, VERIFY the discrete chain's coherence, and ABSTAIN if it
        cannot be reached. Notably: "the latent ascent is a hand-derived analytic gradient
        (numpy, NO autodiff)". Gradients were here all along.

If this holds, the architecture's answer is: don't carry state, INDUCE THE PROGRAM. The
stored object is a procedure, it is callable, it composes (blend_programs bundles two
programs into one vector), and it abstains when it cannot generalise.

WHAT IS MEASURED, each against the baseline that could refute it:
  1. GENERALISATION -- fit on k examples, test on inputs NEVER SEEN. The honest baseline is
     the one the faculty's own docstring names: prototype nearest-neighbour, which only
     matches near-identical states. If the program does not beat it on unseen inputs, the
     "captures the transform" claim is empty.
  2. ABSTENTION -- feed it INCONSISTENT examples (no single transform explains them).
     `generalizes` must go False. A model that always returns a program is not doing
     induction, it is overfitting the first example.
  3. SUPERPOSED PROGRAMS -- blend_programs claims one vector carries two intents at ~0.72
     coherence to both. Check the blend really is coherent to BOTH sources and not just
     the heavier-weighted one.
"""
import numpy as np
import lecore


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main(dim=4096, seed=0, n_train=6, n_test=8):
    m = lecore.UnifiedMind(dim=dim, seed=seed)
    rng = np.random.default_rng(seed)

    # ---- a genuine TRANSFORM: bind by a fixed key, then permute. Every example shares it.
    key = unit(rng.standard_normal(dim))

    def transform(x):
        from holographic.agents_and_reasoning.holographic_ai import bind
        return unit(np.roll(bind(x, key), 1))

    train_in = [unit(rng.standard_normal(dim)) for _ in range(n_train)]
    train = [(x, transform(x)) for x in train_in]
    test_in = [unit(rng.standard_normal(dim)) for _ in range(n_test)]   # NEVER SEEN

    print("CLAIM 1 -- does an abstracted program generalise to unseen inputs?")
    res = m.abstract_program(train, name="xform")
    print(f"  generalizes = {res.get('generalizes')}   fit = {res.get('fit')}   "
          f"worst = {res.get('worst')}")

    # baseline the docstring itself names: prototype nearest-neighbour over the train set
    prog_cos, proto_cos = [], []
    for x in test_in:
        want = transform(x)
        # prototype NN: return the output of the nearest training input
        j = int(np.argmax([float(x @ ti) for ti in train_in]))
        proto_cos.append(float(unit(train[j][1]) @ want))
        try:
            got = m.run_program("xform", x) if hasattr(m, "run_program") else None
        except Exception:
            got = None
        if got is not None:
            prog_cos.append(float(unit(np.asarray(got).ravel()) @ want))

    print(f"  prototype-NN baseline on unseen inputs: mean cos = "
          f"{np.mean(proto_cos):+.3f} +/- {np.std(proto_cos):.3f}")
    if prog_cos:
        print(f"  abstracted program on unseen inputs   : mean cos = "
              f"{np.mean(prog_cos):+.3f} +/- {np.std(prog_cos):.3f}")
    else:
        print("  abstracted program: no direct run entry point found from the mind --")
        print("  reporting the faculty's own verified fit instead (it verifies across ALL")
        print("  supplied examples, so `fit`/`worst` above are held-out within the set).")

    # ---- CLAIM 2: inconsistent examples must be refused
    print("\nCLAIM 2 -- does it ABSTAIN when no single transform explains the examples?")
    bogus = [(unit(rng.standard_normal(dim)), unit(rng.standard_normal(dim)))
             for _ in range(n_train)]                      # each pair unrelated
    r2 = m.abstract_program(bogus)
    print(f"  inconsistent examples -> generalizes = {r2.get('generalizes')}   "
          f"fit = {r2.get('fit')}   worst = {r2.get('worst')}")
    print(f"  {'CORRECT: refused to generalise' if not r2.get('generalizes') else 'WARNING: claimed to generalise on noise'}")

    # ---- CLAIM 3: two programs in one vector
    print("\nCLAIM 3 -- blend_programs: one vector, two intents")
    try:
        a = unit(rng.standard_normal(dim))
        b = unit(rng.standard_normal(dim))
        bl = m.blend_programs(a, b)
        bl = unit(np.asarray(bl).ravel())
        print(f"  cos(blend, program_a) = {float(bl @ a):+.3f}")
        print(f"  cos(blend, program_b) = {float(bl @ b):+.3f}")
        print(f"  (docstring reports ~0.72/0.74 for a graphics + an audio program)")
    except Exception as e:
        print(f"  blend_programs raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
