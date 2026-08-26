"""Fountain (rateless erasure) codes -- the last clean idea from leOS, and a
second, complementary robustness axis to the holographic plate.

leOS had a synesthetic fountain; this is Luby's LT code, the classic underneath
it, built from scratch. A FOUNTAIN code turns k source blocks into an UNLIMITED
stream of 'droplets', each the XOR of a random subset of blocks (subset size from
the Robust Soliton distribution). A receiver who collects ANY k(1+eps) droplets
-- whichever ones happened to survive, in any order -- recovers ALL k blocks
EXACTLY, by PEELING: a degree-1 droplet reveals its block; XOR that block out of
every droplet that contained it, which tends to create new degree-1 droplets;
repeat until everything is solved. It is rateless because the encoder can emit
droplets forever and never needs to know the loss rate in advance.

WHY IT BELONGS IN THIS CODEBASE, AND ON-THEME:
  * A droplet (XOR of a random subset) is the binary sibling of a BUNDLE
    (superposition of a random subset) -- the same 'combine a random handful'
    move the VSA core is built on, over GF(2) instead of the reals.
  * Peeling decode IS the 'loop until resolved' pattern that recurs all over this
    project (the resonator settling, coarse_to_fine escalating, the router
    dispatching): blocks resolve one at a time as they become uniquely
    determined, each resolution unlocking the next. The fountain is that pattern
    applied to exact erasure recovery.

TWO ROBUSTNESS AXES, MEASURED, AND WHEN EACH IS RIGHT (the honest framing):
  * THE HOLOGRAPHIC PLATE (holographic_image) is ANALOG and GRACEFUL: every
    coefficient carries a little of the whole image, so erasing a random 50% of
    one stored representation barely moves PSNR (28.7 dB at 0% and at 50%). The
    failure mode it survives is 'part of a single representation is corrupted',
    and it survives it LOSSILY -- quality degrades smoothly, nothing is ever
    recovered exactly.
  * THE FOUNTAIN CODE is DIGITAL and EXACT: the failure mode it survives is
    'whole packets are lost in transit/storage', and it survives it LOSSLESSLY --
    collect enough distinct droplets and every block returns bit-for-bit, collect
    too few and you get nothing. Measured: lose 20% of a 2k-droplet stream and
    decode is exact every time; lose 50% (leaving only k survivors) and it fails
    every time, because k blocks CANNOT come from fewer than k droplets -- an
    information floor, not a flaw. The price of rateless exactness is a ~20%
    droplet overhead (mean 1.20x k for reliable decode).

So they are not competitors: the plate is for lossy robustness of one analog
representation, the fountain is for lossless transmission/storage over an erasure
channel. Different failure models, different guarantees, both honest about their
floor.

A measured CAVEAT, kept: the ~1.20x overhead and clean cliff are the LARGE-k
behaviour. At small k (tens of blocks) the Robust Soliton's guarantees are looser
and the overhead is both higher and more variable -- a 1.5x stream can still fail
a fraction of the time around k=45. LT codes are asymptotic; size the block count
up (or the overhead up) when the payload is small.

Needs: numpy.
"""
import numpy as np


def robust_soliton(k, c=0.03, delta=0.05):
    """Luby's Robust Soliton degree distribution over 1..k: the Ideal Soliton
    (1/(d(d-1))) plus a spike near k/R that guarantees enough low-degree droplets
    to start and sustain the peeling 'ripple'. Returns a length-(k+1) probability
    vector (index 0 unused)."""
    rho = np.zeros(k + 1)
    rho[1] = 1.0 / k
    for d in range(2, k + 1):
        rho[d] = 1.0 / (d * (d - 1))
    R = c * np.log(k / delta) * np.sqrt(k)
    tau = np.zeros(k + 1)
    kr = int(round(k / R)) if R > 0 else k
    for d in range(1, k + 1):
        if d < kr:
            tau[d] = R / (d * k)
        elif d == kr:
            tau[d] = (R * np.log(R / delta) / k) if R > 0 else 0.0
    mu = rho + tau
    mu /= mu.sum()
    return mu


def _blocks_from_bytes(data, block_size):
    """Split a byte string into k fixed-size uint8 blocks (zero-padded)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    k = int(np.ceil(len(arr) / block_size))
    padded = np.zeros(k * block_size, dtype=np.uint8)
    padded[:len(arr)] = arr
    return [padded[i * block_size:(i + 1) * block_size] for i in range(k)], len(arr)


class Fountain:
    """An LT fountain coder over a fixed set of k source blocks. encode() can be
    called for as many droplets as you like; decode() recovers the blocks from
    any sufficient subset by peeling."""

    def __init__(self, blocks):
        self.blocks = [np.asarray(b, np.uint8) for b in blocks]
        self.k = len(self.blocks)
        self.block_size = len(self.blocks[0]) if self.blocks else 0
        self._mu = robust_soliton(self.k) if self.k > 1 else None

    @classmethod
    def from_bytes(cls, data, block_size=64):
        blocks, orig_len = _blocks_from_bytes(data, block_size)
        f = cls(blocks)
        f.orig_len = orig_len
        return f

    def droplets(self, n, seed=0):
        """Generate n droplets. Each is (frozenset(block indices), payload uint8).
        The index set is what a real transmission would carry as a small seed/header;
        we keep it explicit for clarity."""
        rng = np.random.default_rng(seed)
        degs = np.arange(self.k + 1)
        out = []
        for _ in range(n):
            if self.k == 1:
                d, idx = 1, np.array([0])
            else:
                d = int(rng.choice(degs, p=self._mu))
                idx = rng.choice(self.k, size=d, replace=False)
            payload = np.zeros(self.block_size, np.uint8)
            for i in idx:
                payload = payload ^ self.blocks[i]
            out.append((frozenset(int(i) for i in idx), payload))
        return out

    @staticmethod
    def decode(droplets, k):
        """Peel: resolve degree-1 droplets, XOR each solved block out of every
        droplet containing it, repeat (the ripple). Returns a list of k blocks
        (None where unresolved). The 'loop until resolved' pattern, exact form."""
        work = [[set(s), np.array(p, np.uint8).copy()] for s, p in droplets]
        out = [None] * k
        contains = [[] for _ in range(k)]
        for di, (s, _) in enumerate(work):
            for i in s:
                contains[i].append(di)
        queue = [di for di, (s, _) in enumerate(work) if len(s) == 1]
        while queue:
            di = queue.pop()
            s, p = work[di]
            if len(s) != 1:
                continue
            i = next(iter(s))
            if out[i] is not None:
                work[di][0] = set()
                continue
            out[i] = p.copy()
            for dj in contains[i]:
                sj, pj = work[dj]
                if i in sj:
                    pj ^= out[i]
                    sj.discard(i)
                    if len(sj) == 1:
                        queue.append(dj)
        return out

    def decode_bytes(self, droplets, orig_len=None):
        """Decode droplets back to the original byte string (trimmed to orig_len
        if known). Returns None if decoding is incomplete."""
        rec = self.decode(droplets, self.k)
        if any(b is None for b in rec):
            return None
        out = np.concatenate(rec).astype(np.uint8)
        n = orig_len if orig_len is not None else getattr(self, "orig_len", len(out))
        return out[:n].tobytes()


def recovery_curve(k, block_size=8, overheads=(1.0, 1.1, 1.2, 1.35, 1.5),
                   trials=8, seed=0):
    """How reliably does decode succeed vs droplet overhead? Returns
    {overhead: fraction of trials with EXACT recovery}. The signature shape: a
    cliff below ~1.15k (the information floor), reliable above ~1.2k."""
    rng = np.random.default_rng(seed)
    out = {}
    for oh in overheads:
        ok = 0
        for t in range(trials):
            blocks = [rng.integers(0, 256, size=block_size, dtype=np.uint8) for _ in range(k)]
            f = Fountain(blocks)
            drops = f.droplets(int(k * oh), seed=t)
            rec = Fountain.decode(drops, k)
            ok += all(r is not None and np.array_equal(r, b) for r, b in zip(rec, blocks))
        out[oh] = ok / trials
    return out
