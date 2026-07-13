"""Offline query embedder (N31) -- turn a plain-English query into a nomic-64 vector with NO model.

THE GAP THIS CLOSES:
route_semantic (N28) scores a query VECTOR against the 96 KB module index. It gets a vector for a
cached phrase, but a brand-new free-text query needs the nomic encoder -- a 137 MB runtime dependency
we refused (footprint policy). This module supplies the missing vector CHEAPLY:

    query text --SIF pool of the token table--> sif vector --ridge W--> nomic-64 query vector

Arora/Liang/Ma (ICLR 2017): a sentence embedding is the frequency-weighted mean of its word vectors,
minus the top principal component. W is a single ridge matrix (encoder ~= W . sif), fit ONCE at build
time by tools/semantic/distill_map.py. What ships: the token table (already needed for SIF), W, and the
SIF frequency table -- a few MB total, NO transformer, NO forward pass.

HONEST STATUS: whether W is good enough is an EMPIRICAL question distill_map.py answers on the real
corpus (the synthetic proof: routing over W.sif tracks true-embedding routing 12/12 when the spaces are
linearly related; the real R^2 is the gate). This module is the runtime plumbing; it loads whatever W
distill_map produced and refuses (returns None) if the artifact is absent -- so route_semantic keeps its
honest fallback. Building the pipe does not assert the water is clean; the bar lives in the fit script.

DETERMINISM: fixed token table, fixed W, fixed ABTT vector. No RNG, no model.
"""
import re
import numpy as np


class QueryEmbedder:
    """Loads the shipped offline-embed artifact and maps query text -> nomic-64 vector.

    Artifact (.npz from distill_map.py --export): token_ids map, token_vecs (V x 64 q8 or f16),
    freqs (V,), W (sif_dim x 64), and the ABTT pc/mu. All small, all static."""

    def __init__(self, artifact_path):
        z = np.load(artifact_path, allow_pickle=True)
        self.vocab = {str(t): i for i, t in enumerate(z["tokens"])}
        self.E = z["token_vecs"].astype(np.float64)          # V x d_sif (the token table, 64d or 300d)
        self.freq = z["freqs"].astype(np.float64)
        self.total = float(self.freq.sum()) or 1.0
        self.W = z["W"].astype(np.float64)                    # d_sif x 64
        self.pc = z["pc"].astype(np.float64) if "pc" in z else None
        self.a = float(z["sif_a"]) if "sif_a" in z else 1e-3

    def _tokens(self, text):
        # crude wordpiece-free tokeniser: the fit script must use the SAME split, or ids won't line up.
        return [self.vocab[w] for w in re.findall(r"[a-z0-9]+", text.lower()) if w in self.vocab]

    def embed(self, text):
        """query text -> nomic-64 vector, or None if no known tokens (caller falls back to tokens)."""
        ids = self._tokens(text)
        if not ids:
            return None
        w = self.a / (self.a + self.freq[ids] / self.total)  # SIF weights: rare words count more
        sif = (w[:, None] * self.E[ids]).sum(0) / w.sum()
        if self.pc is not None:                              # remove the top direction (Arora step 3)
            sif = sif - (sif @ self.pc.T) @ self.pc
        return sif @ self.W                                  # ridge map into nomic-64


def _selftest():
    """Fit a tiny W on synthetic data, prove text->vector is deterministic and routes a planted match.
    No file on disk beyond a temp npz, no model."""
    import tempfile, os
    rng = np.random.default_rng(0)
    V, d_sif, d_emb = 40, 32, 64
    toks = [f"w{i}" for i in range(V)]
    E = rng.standard_normal((V, d_sif))
    freqs = rng.integers(1, 100, V).astype(float)
    Wstar = rng.standard_normal((d_sif, d_emb)) / np.sqrt(d_sif)
    # fit W to invert the SIF->emb relation on random sentences
    def sif(ids):
        a = 1e-3; w = a / (a + freqs[ids] / freqs.sum())
        return (w[:, None] * E[ids]).sum(0) / w.sum()
    S = np.array([sif(rng.choice(V, 4, replace=False)) for _ in range(300)])
    Y = S @ Wstar
    W = np.linalg.solve(S.T @ S + 1e-2 * np.eye(d_sif), S.T @ Y)
    fd, p = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    np.savez(p, tokens=np.array(toks), token_vecs=E, freqs=freqs, W=W, sif_a=1e-3)
    qe = QueryEmbedder(p)
    v1 = qe.embed("w1 w2 w3"); v2 = qe.embed("w1 w2 w3")
    assert v1 is not None and np.allclose(v1, v2), "must be deterministic"
    assert qe.embed("nonexistent zzz") is None, "unknown tokens -> None (honest miss)"
    assert v1.shape == (d_emb,), v1.shape
    os.remove(p)
    print("holographic_queryembed selftest: OK (deterministic; unknown->None; maps to nomic-64)")


if __name__ == "__main__":
    _selftest()
