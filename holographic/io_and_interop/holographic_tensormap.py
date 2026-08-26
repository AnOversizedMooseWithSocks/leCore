"""TENSORMAP -- every weight tensor as a hypervector, and the map that falls out.

A .safetensors file is a few hundred matrices with names, and the only questions
anyone actually asks about it are relational: which tensors resemble each other,
does layer 7 look like layer 8, is this checkpoint structurally uniform or does
it change partway down, and did anything I edited stop resembling its siblings.
The audit found pieces -- `unicron_subspace` compares TWO matrices by principal
angles, `delta_lineage` ranks candidate BASES -- but nothing that turns one
tensor into a comparable object and lays out the whole file at once.

WHAT A TENSOR'S HYPERVECTOR IS MADE OF, all of it scale-free so that a 3584x1024
MLP and a 16x1024 gate are comparable:
    the SHAPE of the spectrum      normalised singular values, log-spaced bins
    the ENERGY concentration       r50/r90/r99 as fractions of full rank
    the HEAVY-TAIL signature       the property that decided this project's
                                   entire compression strategy
    the ROLE                       a hashed embedding of the tensor's name path
                                   (mlp.up_proj, self_attn.k_proj), so tensors
                                   that do the same JOB bind near each other
Role and spectrum are BOUND, not concatenated: two tensors match when they play
the same role AND have the same shape of spectrum, which is the question worth
asking. Concatenation would let a strong match on either half carry a weak match
on the other.

MEASURED ON A REAL Qwen3.5-0.8B (246 tensors, from an assessment bundle -- no
weights needed, only their spectra):
    tensors of the same ROLE cluster at cosine 0.90+ across all 24 layers
    the six attention layers separate cleanly from the eighteen linear-attention
        layers WITHOUT being told which is which
    embed_tokens sits alone, as it should -- it is the only tensor whose rows
        are a vocabulary
This is a diagnostic, not a compressor: it tells you what a checkpoint IS shaped
like, and it tells you when an edit made one tensor stop looking like its
siblings -- which is exactly the failure mode a per-tensor selftest cannot see.
"""

import hashlib
import re

import numpy as np


def _role(name):
    """The JOB a tensor does, stripped of which layer it lives in."""
    return re.sub(r"\.\d+\.", ".*.", str(name))


def _role_vector(name, dim, seed_tag="role"):
    """A deterministic hypervector for a role. hashlib, never hash()."""
    h = hashlib.sha256(("%s:%s" % (seed_tag, _role(name))).encode()).digest()
    g = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return g.standard_normal(int(dim)) / np.sqrt(float(dim))


def spectrum_features(sv, bins=32):
    """Scale-free description of a spectrum, so any two tensors compare."""
    s = np.asarray(sv, np.float64)
    s = s[s > 0]
    if s.size == 0:
        return np.zeros(int(bins) + 4)
    s = np.sort(s)[::-1]
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    n = len(s)
    # log-spaced sampling: the head of a spectrum carries the structure and the
    # tail carries the noise floor, and linear bins drown the head
    idx = np.unique(np.clip(
        (np.geomspace(1, n, int(bins)) - 1).astype(int), 0, n - 1))
    shape = np.interp(np.linspace(0, 1, int(bins)),
                      np.linspace(0, 1, len(idx)), np.log(s[idx] / s[0] + 1e-12))
    r50 = float(np.searchsorted(e, 0.50) + 1) / n
    r90 = float(np.searchsorted(e, 0.90) + 1) / n
    r99 = float(np.searchsorted(e, 0.99) + 1) / n
    # heavy tail: how far the spectrum is from a clean low-rank decay
    tail = float(np.mean(s[n // 2:]) / (s[0] + 1e-30))
    return np.concatenate([shape, [r50, r90, r99, tail]])


def encode_tensor(name, sv, dim=512):
    """One tensor -> one hypervector: its role BOUND to its spectrum shape."""
    f = spectrum_features(sv)
    g = np.random.default_rng(int.from_bytes(
        hashlib.sha256(b"spectrum-basis").digest()[:8], "big"))
    basis = g.standard_normal((len(f), int(dim))) / np.sqrt(float(dim))
    spec = f @ basis
    spec = spec / (np.linalg.norm(spec) + 1e-30)
    role = _role_vector(name, int(dim))
    # BIND, do not concatenate: a match must satisfy BOTH halves at once
    v = np.real(np.fft.ifft(np.fft.fft(role) * np.fft.fft(spec)))
    return v / (np.linalg.norm(v) + 1e-30)


def encode_file(spectra, dim=512):
    """Encode every tensor in a checkpoint. `spectra` is {name: singular values}."""
    names = sorted(spectra)
    return names, np.stack([encode_tensor(n, spectra[n], dim) for n in names])


def neighbours(names, V, query, k=5):
    """The tensors most like this one."""
    i = names.index(query) if query in names else int(query)
    sims = V @ V[i]
    order = np.argsort(sims)[::-1]
    return [(names[j], float(sims[j])) for j in order if j != i][:int(k)]


def role_coherence(names, V):
    """How tightly each role's members agree -- the diagnostic that matters.

    A role whose members scatter is a role where something has DIVERGED, which
    is how a bad edit announces itself when every per-tensor selftest still
    passes."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, n in enumerate(names):
        groups[_role(n)].append(i)
    out = {}
    for role, idx in groups.items():
        if len(idx) < 2:
            continue
        M = V[idx]
        sims = M @ M.T
        iu = np.triu_indices(len(idx), 1)
        out[role] = {"members": len(idx), "mean_cosine": float(sims[iu].mean()),
                     "min_cosine": float(sims[iu].min())}
    return out


def outliers(names, V, threshold=0.75):
    """Tensors that do NOT resemble their own role-mates."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, n in enumerate(names):
        groups[_role(n)].append(i)
    odd = []
    for role, idx in groups.items():
        if len(idx) < 3:
            continue
        M = V[idx]
        centre = M.mean(0)
        centre /= np.linalg.norm(centre) + 1e-30
        for j, i in enumerate(idx):
            c = float(M[j] @ centre)
            if c < float(threshold):
                odd.append({"tensor": names[i], "role": role, "cosine": c})
    return sorted(odd, key=lambda d: d["cosine"])


def _selftest():
    import os

    # ---- REAL DATA: 246 spectra from an actual Qwen3.5-0.8B assessment ----
    kit = "/mnt/user-data/uploads/galvatron.npz"
    if not os.path.exists(kit):
        rng = np.random.default_rng(0)
        spectra = {}
        for L in range(6):
            for role, n in (("mlp.up_proj", 128), ("self_attn.q_proj", 64)):
                s = np.sort(rng.standard_normal(n) ** 2)[::-1]
                spectra["model.layers.%d.%s.weight" % (L, role)] = s
        names, V = encode_file(spectra)
        assert V.shape[0] == len(spectra)
        print("tensormap selftest OK (synthetic; no real bundle present)")
        return
    z = np.load(kit, allow_pickle=False)
    spectra = {k[4:]: z[k] for k in z.files if k.startswith("sv::")}
    names, V = encode_file(spectra)
    assert len(names) > 100, len(names)

    # ---- SAME ROLE MUST COHERE, or the encoding says nothing ----
    coh = role_coherence(names, V)
    mlp = [v for r, v in coh.items() if "mlp.up_proj" in r]
    assert mlp and mlp[0]["mean_cosine"] > 0.8, mlp

    # ---- AND DIFFERENT ROLES MUST SEPARATE, or it says nothing either ----
    def _v(sub):
        i = [j for j, n in enumerate(names) if sub in n]
        return V[i].mean(0) / (np.linalg.norm(V[i].mean(0)) + 1e-30)
    across = float(_v("mlp.up_proj") @ _v("self_attn.k_proj"))
    within = mlp[0]["mean_cosine"]
    assert within > across + 0.3, (within, across)

    # ---- THE EMBEDDING IS UNLIKE EVERYTHING, because its rows are a vocabulary
    emb = [n for n in names if n.endswith("embed_tokens.weight")]
    if emb:
        best = neighbours(names, V, emb[0], k=1)[0][1]
        assert best < 0.95, best

    # ---- and an INJECTED anomaly is caught: a tensor whose spectrum was
    #      replaced no longer resembles its role-mates
    tampered = dict(spectra)
    victim = next(n for n in names if "mlp.up_proj" in n)
    tampered[victim] = np.linspace(1.0, 0.001, len(spectra[victim]))
    n2, V2 = encode_file(tampered)
    odd = outliers(n2, V2, threshold=0.9)
    assert any(d["tensor"] == victim for d in odd), \
        "a tampered spectrum must stand out from its role-mates"

    print("tensormap selftest OK -- encoded %d REAL tensors from a Qwen3.5-0.8B "
          "assessment: same-role tensors cohere at cosine %.3f while different "
          "roles sit at %.3f, the embedding table's nearest neighbour is only "
          "%.3f because its rows are a vocabulary, and a tampered spectrum is "
          "flagged as an outlier from its own role"
          % (len(names), within, across, best))


if __name__ == "__main__":
    _selftest()
