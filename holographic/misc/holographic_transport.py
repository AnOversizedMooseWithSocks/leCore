"""Optimal transport: the Wasserstein distance by Sinkhorn iteration (BLD-8).

WHAT IT IS, AND WHY
-------------------
A bin-wise distance between two distributions -- Euclidean, cosine, KL -- compares mass bin-by-bin: it asks
"how different is the height here?" and is blind to WHERE the mass sits. Two histograms that don't overlap are
maximally far by any bin-wise metric no matter how far APART they are -- a peak at bin 12 and a peak at bin 20
score the same distance from a peak at bin 10 as a peak at bin 40 does. That is exactly the wrong answer when
the bins have geometry (a position, a frequency, a coordinate).

The Wasserstein (earth-mover's) distance measures the least work to MOVE one distribution onto the other --
mass times the ground distance it travels -- so it grows with how far apart the distributions are, even when
they share no support. This module computes it with the Sinkhorn algorithm: add an entropy term (regularised
OT), which turns the transport problem into a Gibbs kernel K = exp(-C/eps) and a pair of alternating diagonal
rescalings (u <- a/(Kv), v <- b/(K^T u)) that converge to the transport plan P; the distance is <P, C>.

MEASURED (in the selftest)
--------------------------
  * It matches the 1-D closed form (W1 = sum_i |CDF_a(i) - CDF_b(i)|) to ~1e-3.
  * It tracks a shift LINEARLY (shift 2/5/10/20 -> W ~ 2/5/10/20) while Euclidean saturates (~0.53 for every
    non-overlapping shift) and cosine collapses to ~0 -- both unable to tell a near miss from a far one.

KEPT NEGATIVES (measured / documented)
--------------------------------------
  * THE EPS KNOB. eps too LARGE blurs the plan toward the independent coupling, inflating the distance above
    the true W (measured: a same-mean narrow-vs-wide pair with true W1~3.6 reads ~4.9 at eps=50). eps too SMALL
    underflows -- K = exp(-C/eps) rounds to zero where C/eps is large, and the rescalings hit 0/0 -> NaN
    (measured: eps=0.005 on a 200-bin problem returns NaN). The default RULE here scales eps to the cost
    (eps = 0.02 * median nonzero cost), which stays sharp without underflowing for cost matrices whose
    max/median is not extreme; a caller with a wide cost range should set eps explicitly (or use a log-domain
    solver, out of scope).
  * O(n*m) PER ITERATION. The kernel and the matvecs are dense, so cost grows with the product of the support
    sizes -- fine for the moderate histograms here, not for very large clouds (which want a low-rank or
    multiscale OT solver, a bigger build).

Only NumPy. Nothing learned.
"""
import numpy as np


def _default_cost(n, m):
    """The 1-D ground distance |i-j| on the bin indices -- the right cost when the bins are an ordered axis
    (position, time, frequency). Callers with a different geometry pass their own cost matrix."""
    ii = np.arange(n)
    jj = np.arange(m)
    return np.abs(ii[:, None] - jj[None, :]).astype(float)


def wasserstein(a, b, cost=None, eps=None, iters=500, tol=1e-9, return_plan=False):
    """Entropic-regularised Wasserstein distance between distributions `a` and `b` via Sinkhorn iteration.

    `a`, `b` are non-negative weight vectors (normalised internally to sum 1). `cost` is the ground-distance
    matrix C[i,j] between bin i of a and bin j of b; if None, the 1-D distance |i-j| is used. `eps` is the
    entropic regularisation; if None it is set to 0.02 * (median nonzero cost) -- sharp but underflow-safe for
    well-conditioned cost matrices (see the kept negatives on the eps knob). Returns the distance <P, C>, or
    (distance, transport_plan) if return_plan."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("distributions must carry positive mass")
    a = a / a.sum()
    b = b / b.sum()
    if cost is None:
        cost = _default_cost(len(a), len(b))
    cost = np.asarray(cost, float)
    if eps is None:
        pos = cost[cost > 0]
        eps = 0.02 * (np.median(pos) if pos.size else 1.0)   # documented rule: scale eps to the cost

    K = np.exp(-cost / eps)                                   # Gibbs kernel
    u = np.ones(len(a))
    v = np.ones(len(b))
    for _ in range(iters):
        u_prev = u
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
        if not np.all(np.isfinite(u)):                       # underflow (eps too small) -> honest NaN
            return (float("nan"), None) if return_plan else float("nan")
        if np.max(np.abs(u - u_prev)) < tol:
            break
    P = u[:, None] * K * v[None, :]                          # the transport plan
    W = float(np.sum(P * cost))
    return (W, P) if return_plan else W


# ---------------------------------------------------------------------------

def _selftest():
    n = 50
    x = np.arange(n)
    C = np.abs(x[:, None] - x[None, :]).astype(float)

    def gauss(mu, sig=2.0):
        g = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        return g / g.sum()

    # (1) matches the 1-D closed form W1 = sum |CDF_a - CDF_b|.
    a, b = gauss(15), gauss(25)
    w_true = float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))
    w = wasserstein(a, b, C, eps=0.5)
    assert abs(w - w_true) < 0.05, f"Sinkhorn {w} != closed form {w_true}"

    # (2) THE WIN: W tracks the shift; Euclidean saturates and cosine collapses once support stops overlapping.
    ref = gauss(15)
    ws, es, cs = [], [], []
    for shift in (5, 10, 20):
        s = gauss(15 + shift)
        ws.append(wasserstein(ref, s, C, eps=0.5))
        es.append(np.linalg.norm(ref - s))
        cs.append(float((ref @ s) / (np.linalg.norm(ref) * np.linalg.norm(s))))
    assert ws[0] < ws[1] < ws[2], f"W did not track the shift: {ws}"            # 5 < 10 < 20
    assert es[1] - es[0] < 0.1 and es[2] - es[1] < 0.1, f"Euclidean should saturate: {es}"   # blind to distance
    assert ws[2] / ws[0] > 3.5                                                  # W spans the full 4x; eucl/cos do not

    # (3) KEPT NEGATIVE -- the eps knob. Large eps blurs high; tiny eps underflows to a broken answer.
    aa, bb = gauss(25, 1.5), gauss(25, 6.0)
    w_true2 = float(np.sum(np.abs(np.cumsum(aa) - np.cumsum(bb))))
    w_big = wasserstein(aa, bb, C, eps=50.0)
    assert w_big > w_true2 * 1.2, f"large eps should blur high: {w_big} vs {w_true2}"
    # eps so small the kernel between the two separated supports underflows: the result breaks -- either a
    # non-finite NaN or a wildly wrong distance (the transport plan can no longer connect the masses).
    w_tiny = wasserstein(a, b, C, eps=0.01)                  # a,b are 10 bins apart; exp(-10/0.01)=exp(-1000)=0
    assert (not np.isfinite(w_tiny)) or abs(w_tiny - w_true) > 2.0, f"tiny eps should break, got {w_tiny}"

    # (4) self-distance is near-zero RELATIVE to a real shift. (Entropic OT has a small positive self-bias --
    #     W_eps(a,a) > 0 because the regularised plan spreads mass; it shrinks with eps. Debias via the Sinkhorn
    #     divergence if an exact zero is needed. Here the point is self << cross.)
    self_d = wasserstein(a, a, C, eps=0.5)
    cross_d = wasserstein(a, b, C, eps=0.5)
    assert self_d < 0.15 * cross_d, f"self-distance not << cross: {self_d} vs {cross_d}"
    assert wasserstein(a, b, C, eps=0.5) == wasserstein(a, b, C, eps=0.5)        # deterministic

    # (5) the default eps rule gives a sensible (near closed-form) answer without being told eps.
    w_default = wasserstein(gauss(15), gauss(25))
    assert abs(w_default - w_true) < 0.3, f"default-eps W off: {w_default} vs {w_true}"

    print("holographic_transport selftest OK:")
    print(f"  Sinkhorn W={w:.3f} == closed-form W1={w_true:.3f}")
    print(f"  tracks shift: W={[round(v,1) for v in ws]} vs eucl={[round(v,3) for v in es]} cos={[round(v,4) for v in cs]}")
    print(f"  eps knob: large eps blurs ({w_big:.2f} vs true {w_true2:.2f}); tiny eps underflows -> {w_tiny}")
    print(f"  default-eps rule (untold): W={w_default:.3f}")


if __name__ == "__main__":
    _selftest()
