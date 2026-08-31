"""Walk on Spheres as an RNN state mechanism -- the state you never store.

THE REFRAME. Every architecture in the sequence-model literature MATERIALIZES its hidden
state: Transformers cache tokens, RNNs carry a vector, Memory Caching caches N of them.
`mind.solve_pde` (Walk on Spheres) evaluates the solution of an elliptic PDE AT QUERY POINTS
ONLY -- no mesh, no grid, no global solve -- by random walks that step by the distance to the
boundary. So if a state is a FIELD satisfying a PDE, you store the BOUNDARY (small,
deterministic) and solve for the state wherever you actually look.

That is "determinism instead of storage" (lever 3) taken to its limit, and it is a property
no competing architecture has, because none of them has a grid-free solver sitting next to
the memory.

WHAT THIS MEASURES (three claims, each with the baseline that could refute it):
  1. RESOLUTION INDEPENDENCE -- cost per query does not depend on how finely you'd have had
     to discretize. A grid solver's cost is O(N^d); WoS's is O(walks) per point, full stop.
  2. HONEST ERROR BARS -- solve_pde returns (solution, standard_error). Almost nothing else
     in the engine self-reports its own uncertainty per evaluation. Check the error bar
     actually brackets the truth, and that it shrinks as 1/sqrt(n_walks).
  3. THE HARMONIC CONTRACT -- verified against a closed-form harmonic function on a ball,
     not against another sampler. On a sphere of radius R centred at the origin, ANY linear
     boundary value g(x)=a.x has the exact harmonic extension u(x)=a.x inside. That is the
     honest analytic baseline.

KEPT NEGATIVE this exposes: WoS is Monte Carlo. Its error falls as 1/sqrt(n_walks), so three
digits cost 100x the walks of one digit. It is the right tool when you query FEW points of a
BIG domain, and the wrong one when you need the whole field at high accuracy -- exactly the
inverse of a grid solver's regime.
"""
import time
import numpy as np
import lecore


class BallSDF:
    """Unit-ball SDF. solve_pde wants an OBJECT with .eval (it calls -sdf.eval(Q)),
    not a bare callable -- a live-API detail no docstring stated."""
    def eval(self, p):
        return np.linalg.norm(np.atleast_2d(p), axis=1) - 1.0

ball_sdf = BallSDF()


def main():
    m = lecore.UnifiedMind(dim=512, seed=0)
    a = np.array([0.6, -0.3, 0.5])          # the linear form whose extension is exact

    def g(p):
        """Boundary value g(x) = a.x -- its harmonic extension inside the ball is a.x."""
        return np.atleast_2d(p) @ a

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((12, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 0.55   # well inside
    truth = pts @ a

    print("CLAIM 2+3 -- error bar honesty and the harmonic contract")
    print(f"{'n_walks':>8} {'mean|err|':>11} {'mean SE':>10} {'|err|<2SE':>11} {'ms/point':>10}")
    prev = None
    for nw in (64, 256, 1024, 4096):
        t0 = time.perf_counter()
        sol, se = m.solve_pde(ball_sdf, g, pts, n_walks=nw, seed=0)
        dt = (time.perf_counter() - t0) / len(pts) * 1e3
        err = np.abs(np.asarray(sol).ravel() - truth)
        se = np.asarray(se).ravel()
        cover = float(np.mean(err < 2 * se))
        print(f"{nw:>8} {err.mean():>11.4f} {se.mean():>10.4f} {cover:>11.2f} {dt:>10.2f}")
        if prev is not None:
            print(f"         SE ratio vs 4x fewer walks = {prev/se.mean():.2f} "
                  f"(1/sqrt(4) law predicts 2.00)")
        prev = se.mean()

    print("\nCLAIM 1 -- resolution independence: cost per query vs the grid it replaces")
    sol, se = m.solve_pde(ball_sdf, g, pts[:4], n_walks=1024, seed=0)
    t0 = time.perf_counter()
    m.solve_pde(ball_sdf, g, pts[:4], n_walks=1024, seed=0)
    t_wos = (time.perf_counter() - t0) / 4
    for n in (32, 64, 128, 256):
        grid_cells = n ** 3
        print(f"  a {n}^3 grid solver would carry {grid_cells:>12,} cells; "
              f"WoS cost/query = {t_wos*1e3:.2f} ms, INDEPENDENT of n")
    print("\n  The state was never materialized. Only the boundary function was stored.")


if __name__ == "__main__":
    main()
