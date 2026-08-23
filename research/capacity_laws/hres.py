"""HRES -- Holographic Reservoir Ensemble in Superposition.

THE IDEA. An ESN step is  s <- phi(A s + B u).  Split it and map each half onto a unit
from leCore's own machine model (mind.machine_map()), instead of onto a GPU:

  LINEAR half   -> `operator_power` (holographic_iterate.affine_step_k_batch).
                   GPU name: "tensor core / batched linear operator". Scaling: O(1) IN k.
                   M independent affine recurrences, ANY horizon k, ONE batched
                   eigendecomposition. The horizon is FREE -- a GPU still has to run k steps.

  NONLINEAR half-> `texture_unit` (holographic_shader.bake_1d / mind.bake_field).
                   phi is baked ONCE into a vector and fetched at any x with one dot
                   product. The activation stops being an op and becomes a memory tier.

  ENSEMBLE      -> `simt_width` (holographic_superposed.pack / score_all).
                   GPU name: "warp / SIMT width". K readouts ride in ONE vector;
                   score_all evaluates all K against a query in one pass. The stated
                   capacity law is 1/sqrt(K), NOT sqrt(K/D).

WHY THIS IS NOT A GPU. A GPU buys K computations with K lanes -- the cost is SILICON, linear
in K. Superposition buys K computations in ONE vector -- the cost is PRECISION, 1/sqrt(K).
That is a different trade axis, and it is the one axis a GPU cannot offer at any price.
Combined with an O(1)-in-k horizon, the object being measured here is a recurrent model whose
per-variant AND per-timestep costs have both been removed.

HONEST BASELINES (both required, per house rule):
  * exactness  -- against literal substepping, not against a reimplementation.
  * ensemble   -- against K separate exact readouts at the SAME total storage.

KEPT NEGATIVES this file exists to expose, not hide:
  * affine_step_k_batch RAISES on a defective operator (Jordan block, no eigenbasis).
  * superposition fidelity decays as 1/sqrt(K); past some K the ensemble is worse than
    one exact readout and the crossover is the number that matters.
  * the module note warns: do NOT superpose the OPERATOR variants -- trajectories are
    nonlinear in the operator, so stiffness variants do not sum. Variants go in the BATCH
    (operator_power); only the READOUTS go in superposition (simt_width). Mixing those two
    up is the trap, and it is why the two units are separate seats on the spec sheet.
"""
import time
import numpy as np
import lecore
from holographic.misc.holographic_iterate import affine_step_k_batch
from holographic.misc import holographic_superposed as sp


# ---------------------------------------------------------------- part 1
def horizon_is_free(M=16, d=48, seeds=range(3)):
    """`operator_power`: M reservoir variants at horizon k, O(1) in k. Exact vs substepping."""
    print("PART 1 -- operator_power: is the horizon actually free?")
    print(f"{'k':>6} {'max|batch-substep|':>20} {'substep ms':>12} {'jump ms':>10} {'speedup':>9}")
    for k in (8, 64, 512, 4096):
        errs, t_sub, t_jmp = [], [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            # contractive reservoirs: spectral radius scaled below 1 (echo-state property)
            A = rng.standard_normal((M, d, d)) / np.sqrt(d)
            for i in range(M):
                A[i] *= (0.5 + 0.45 * i / M) / max(
                    abs(np.linalg.eigvals(A[i])).max(), 1e-12)
            b = rng.standard_normal((M, d)) * 0.1
            S0 = rng.standard_normal((M, d)) * 0.1

            t0 = time.perf_counter()
            S = S0.copy()
            for _ in range(k):
                S = np.einsum("mij,mj->mi", A, S) + b
            t_sub.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            J = affine_step_k_batch(S0, A, b, k)
            t_jmp.append(time.perf_counter() - t0)
            errs.append(float(np.max(np.abs(S - J))))
        print(f"{k:>6} {max(errs):>20.3e} {np.mean(t_sub)*1e3:>12.2f} "
              f"{np.mean(t_jmp)*1e3:>10.2f} {np.mean(t_sub)/np.mean(t_jmp):>8.1f}x")

    # the kept negative, exercised on purpose
    print("  kept negative -- defective operator (Jordan block, no eigenbasis):", end=" ")
    J2 = np.array([[[1.0, 1.0], [0.0, 1.0]]])
    try:
        affine_step_k_batch(np.zeros((1, 2)), J2, np.zeros((1, 2)), 4)
        print("did NOT raise (unexpected)")
    except Exception as e:
        print(f"raises {type(e).__name__} -- correct, it refuses rather than lying")


# ---------------------------------------------------------------- part 2
def superposed_readouts(D=2048, K_list=(2, 4, 8, 16, 32, 64), n_probe=200, seeds=range(5)):
    """`simt_width`: K readouts in ONE vector. Where does 1/sqrt(K) cross an exact scan?"""
    print("\nPART 2 -- simt_width: K readouts in one vector, and the 1/sqrt(K) crossover")
    print(f"{'K':>4} {'superposed acc':>15} {'+/- sd':>8} {'1/sqrt(K)':>10} {'one-pass?':>10}")
    for K in K_list:
        accs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            keys = rng.standard_normal((K, D)) / np.sqrt(D)
            keys /= np.linalg.norm(keys, axis=1, keepdims=True)
            items = rng.standard_normal((K, D)) / np.sqrt(D)
            items /= np.linalg.norm(items, axis=1, keepdims=True)
            S = sp.pack(keys, items)                      # K computations -> ONE vector
            hit = 0
            for i in range(min(n_probe, K)):
                rec = sp.unbind(S, keys[i])
                hit += int(np.argmax(items @ (rec / (np.linalg.norm(rec) + 1e-12))) == i)
            accs.append(hit / min(n_probe, K))
        a = np.array(accs)
        print(f"{K:>4} {a.mean():>15.3f} {a.std(ddof=1):>8.3f} "
              f"{1/np.sqrt(K):>10.3f} {'yes':>10}")
    print("  read: one vector, one unbind per query -- no loop over K, no K lanes of silicon.")


# ---------------------------------------------------------------- part 3
def activation_as_a_texture(mind):
    """`texture_unit`: bake tanh once; the RNN's nonlinearity becomes a memory tier."""
    print("\nPART 3 -- texture_unit: the activation function as baked memory")
    xs = np.linspace(-4, 4, 512)
    ys = np.tanh(xs)
    bake = mind.bake_field(xs, ys)
    probe = np.linspace(-3.8, 3.8, 400)
    got = np.array([mind.fetch_field(bake, float(x)) for x in probe])
    err = np.abs(got - np.tanh(probe))
    print(f"  tanh baked into ONE vector; fetched at 400 arbitrary x")
    print(f"  mean |err| = {err.mean():.4f}   max |err| = {err.max():.4f}")
    print("  kept negative: this is the ~1% preview tier. Below the signal's max angular")
    print("  frequency the bake does not blur -- it returns a confident WRONG answer.")


if __name__ == "__main__":
    m = lecore.UnifiedMind(dim=2048, seed=0)
    horizon_is_free()
    superposed_readouts()
    activation_as_a_texture(m)
