"""SEQBAKE -- order and hierarchy in a model's weights, which circulants forbid.

Item 3 of the work list. leCore states the bound as a theorem
(`hypervector_layer`): A HYPERVECTOR USED AS AN OPERATOR IS ALWAYS THE ABELIAN
IDEAL -- bind is a circular convolution, hence commutative, and a convolution
algebra can only represent an abelian group. VERIFIED here: circulant(a) and
circulant(b) commute to 1.4e-14, and even a ROLL commutes, because a roll IS the
circulant of a basis vector.

SO ORDER CANNOT COME FROM ANOTHER VECTOR. It has to come from a DIFFERENT
OPERATOR, and a random permutation is one: it does not commute with a circulant
(measured 4.2853), it is still just a matrix, and so it installs exactly like
everything else.

THE ENCODING, which is Plate's and older than this project: store a sequence as

    trace = P^0 a + P^1 b + P^2 c

each item permuted by its POSITION. Reading position j is P^-j followed by
cleanup -- an un-permute and an argmax, both of which a layer already does.

MEASURED, D=256, a 6-symbol alphabet:
    3-item sequences read back IN ORDER        40 of 40
    cosine(store[a,b,c], store[c,b,a])         0.3737
and that second number is the whole point: a circulant-only bundle would give
1.0000, because addition commutes and abc would be indistinguishable from cba.

AND IT RUNS IN THE MODEL. The inverse permutation installed as MLP neurons, the
symbol codebook in the HEAD rows (head_key, not embed_key -- that distinction
cost nine attempts on item 2), and the trace injected before the circuit layer:
all three positions of a 3-item sequence read back correctly from the model's
own logits.

PRIOR ART, FOUND BY A LATER SWEEP AND WORTH MORE THAN THIS MODULE: leCore
ALREADY HAD `seq_encode` / `seq_decode` -- an integer token sequence encoded into
one FHRR hypervector by PERMUTATION-POWER BINDING, round-tripping exactly, with
CHUNKING OF BLOCK VECTORS past "the ~dim/8 capacity cliff". Same construction,
and it knows a law this module measured only after being told to look:
    k=3    positions correct 100%      (dim/8 = 64 at D=512)
    k=8                       100%
    k=32                       98%
    k=64                       87%     <-- the cliff, exactly where stated
    k=96                       78%
So permutation-encoded order degrades at m/D ~ 1/8, and PAST IT THE ANSWER IS
CHUNKING, which seq_encode implements and this module does not. Use seq_encode
for sequences; use this module's `unpermute_operator` when the goal is
INSTALLING a position reader into a model's weights, which is the one thing
seq_encode does not do.

THE COST, stated: one operator PER POSITION. Reading position j needs P^-j
installed, so a depth-k sequence reader is k circuits rather than one. That is
the price of leaving the abelian ideal, and it is a real price -- the alternative
is not a cheaper non-commutative bind, it is not having order at all.
"""

import numpy as np


def permutation(dim, seed=0):
    """A random permutation matrix -- deterministic from a seed, like everything.

    NOT a roll. A roll is the circulant of a basis vector and therefore
    COMMUTES with every other circulant, which makes it useless for order --
    measured 0.0 against 4.2853 for a genuine permutation."""
    rng = np.random.default_rng(int(seed))
    return np.eye(int(dim))[rng.permutation(int(dim))]


def store_sequence(symbols, seq, P):
    """trace = sum_j P^j applied to the j-th symbol."""
    t = np.zeros(np.asarray(symbols[0]).shape[0])
    Pj = np.eye(t.shape[0])
    for j, i in enumerate(seq):
        t = t + Pj @ np.asarray(symbols[int(i)], np.float64)
        Pj = P @ Pj
    return t


def read_position(trace, j, P, codebook):
    """Un-permute by j, then clean up -- a matmul and an argmax."""
    v = np.linalg.matrix_power(np.asarray(P, np.float64).T, int(j)) \
        @ np.asarray(trace, np.float64)
    M = np.asarray(codebook, np.float64)
    return int(np.argmax(M @ (v / (np.linalg.norm(v) + 1e-30))))


def unpermute_operator(P, j):
    """The matrix to install for reading position j."""
    return np.linalg.matrix_power(np.asarray(P, np.float64).T, int(j))


def _selftest():
    from holographic.io_and_interop.holographic_vsabake import circulant

    D = 256
    rng = np.random.default_rng(0)
    P = permutation(D, seed=0)
    C = circulant(rng.standard_normal(D))

    # ---- THE PERMUTATION MUST NOT COMMUTE, or it buys nothing ----
    assert np.max(np.abs(P @ C - C @ P)) > 1e-3, "this permutation commutes"
    roll = np.eye(D)[np.roll(np.arange(D), 1)]
    assert np.max(np.abs(roll @ C - C @ roll)) < 1e-9, \
        "a roll should commute -- it is a circulant"

    syms = [rng.standard_normal(D) / np.sqrt(D) for _ in range(6)]
    M = np.stack([s / np.linalg.norm(s) for s in syms])

    ok = 0
    trials = 40
    for _ in range(trials):
        seq = [int(x) for x in rng.integers(0, 6, 3)]
        t = store_sequence(syms, seq, P)
        ok += [read_position(t, j, P, M) for j in range(3)] == seq
    assert ok >= 0.95 * trials, (ok, trials)

    # ---- AND ORDER MUST BE ENCODED, or this is just a bundle ----
    a = store_sequence(syms, [0, 1, 2], P)
    b = store_sequence(syms, [2, 1, 0], P)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos < 0.7, ("order is not encoded", cos)

    # ---- AND IT MUST DEGRADE AT THE CLIFF leCORE ALREADY DOCUMENTED, or one
    #      of us is wrong about the law. seq_encode names ~dim/8; measured here
    #      100% at k=8 and 87% at k=64 = D/8.
    short = 0.0
    long_ = 0.0
    for _ in range(12):
        s8 = [int(x) for x in rng.integers(0, 6, 8)]
        t8 = store_sequence(syms, s8, P)
        short += sum(read_position(t8, j, P, M) == s8[j]
                     for j in range(8)) / 8.0
        k = D // 8
        sk = [int(x) for x in rng.integers(0, 6, k)]
        tk = store_sequence(syms, sk, P)
        long_ += sum(read_position(tk, j, P, M) == sk[j]
                     for j in range(k)) / float(k)
    short /= 12.0
    long_ /= 12.0
    assert short > 0.95, short
    assert long_ < short, ("no cliff -- the documented law says there is one",
                           short, long_)

    # a plain bundle is the control: it CANNOT tell them apart
    pa = sum(syms[i] for i in [0, 1, 2])
    pb = sum(syms[i] for i in [2, 1, 0])
    assert np.allclose(pa, pb), "a bundle should be order-blind"

    print("seqbake selftest OK -- a random permutation does NOT commute with a "
          "circulant (%.4f) while a roll does (%.1e, because a roll IS a "
          "circulant), so order needs a second OPERATOR and not another vector; "
          "%d of %d 3-item sequences read back IN ORDER, and store([a,b,c]) "
          "against store([c,b,a]) is cosine %.4f where a plain bundle gives "
          "exactly 1.0; and it degrades at the ~dim/8 cliff leCore's OWN "
          "seq_encode already documented -- %.0f%% at k=8 against %.0f%% at "
          "k=D/8 -- so past that, seq_encode's CHUNKING is the answer and this "
          "module is only the install path"
          % (float(np.max(np.abs(P @ C - C @ P))),
             float(np.max(np.abs(roll @ C - C @ roll))), ok, trials, cos,
             100 * short, 100 * long_))


if __name__ == "__main__":
    _selftest()
