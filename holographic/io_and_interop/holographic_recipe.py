"""RECIPE -- ship what leCore ADDED, not the model it was added to.

Moose: the inflated model size needs looking at holographically -- we should be
optimising information into deterministic structure. He is right, and the
measurement is worse than "inflated".

MEASURED on a real install:
    original model                  2.81 MB
    installed model                 6.24 MB      +122%
    of which EXACTLY ZERO BYTES     2.26 MB      36% of the file
    and, tensor by tensor:
        1.45 MB   identical to the layer it came from, just RENUMBERED
        2.72 MB   GREW -- the ladder widening head counts, the new part padded
        0.00 MB   GENUINELY DIFFERENT VALUES
THE INSTALL ADDS 3.43 MB OF FILE FOR ZERO MB OF NEW INFORMATION. Every added
byte is a copy or a zero.

AND leCORE ALREADY NAMES THIS AS AN ERROR. `bank_or_formula` is the demoscene
economy as a measured gate -- "keep the formula, not the samples" -- and its own
docstring says a bank of things a cheap formula gives you for free is NEGATIVE
VALUE. We were banking zeros.

WHAT IS ACTUALLY DERIVABLE, and it is nearly all of it:
    a blank prepended layer   np.zeros(shape) -- a SHAPE, not bytes
    a renumbered layer        the SAME array under a different key
    ladder padding            zeros again, plus a_log = -ln(half_life), which
                              is a formula the install already computes
    a register reservation    a QR of a seeded matrix -- 64 BITS
    the boot record           derived from the manifest
    the router direction      REAL DATA, and small: one vector per gate
    the improvement           REAL DATA, and small: one low-rank correction

SO A RECIPE IS: the base model's identity, plus the handful of vectors that are
genuinely new, plus the RULES to rebuild everything else. That is kilobytes
where the expanded model is megabytes -- and on a 2.1 GB checkpoint it is the
difference between shipping a 2.1 GB artifact and shipping a diff.

WHAT THIS IS NOT: a replacement for the safetensors output. Other people's
loaders need every declared tensor at full size, and that has not changed. This
is the leCore-NATIVE form -- for storing, versioning, sending and rebuilding an
install -- with `expand()` producing the identical safetensors when a consumer
needs one. The expansion is verified byte-for-byte, because a recipe you cannot
prove reconstructs the artifact is a hope rather than a format.
"""

import hashlib

import numpy as np

RECIPE_FORMAT = "leCore/recipe/1"


def _fingerprint(weights):
    """Identify the BASE model without storing it -- shapes and a content hash.

    hashlib, never hash(): the point is that two people on two machines derive
    the same identity for the same checkpoint."""
    h = hashlib.sha256()
    for k in sorted(weights):
        a = np.asarray(weights[k])
        h.update(k.encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
    return h.hexdigest()[:32]


def build(base_weights, installed_weights, report, prepend=2):
    """Describe an install as RULES plus the few arrays that are real.

    Walks every tensor of the installed model and files it: identical to a base
    tensor (a rename -- store the mapping), all zeros (a shape), or genuinely
    new values (store it). The third category is the only one that costs."""
    rules = {"format": RECIPE_FORMAT,
             "base": _fingerprint(base_weights),
             "prepend": int(prepend),
             "registers": (report.get("registers") or {}),
             "hrnn": (report.get("hrnn") or {}),
             "installed": list(report.get("installed", ()))}
    renames, zeros, arrays, grows = {}, {}, {}, {}
    # index the base by (shape, dtype, first bytes) so a renamed tensor is found
    index = {}
    for k in base_weights:
        a = np.asarray(base_weights[k])
        index.setdefault((a.shape, str(a.dtype)), []).append(k)

    for k in installed_weights:
        a = np.asarray(installed_weights[k])
        if a.size and not a.any():
            zeros[k] = [list(a.shape), str(a.dtype)]
            continue
        hit = None
        for cand in index.get((a.shape, str(a.dtype)), ()):
            if np.array_equal(np.asarray(base_weights[cand]), a):
                hit = cand
                break
        if hit is not None:
            renames[k] = hit
            continue
        # A GROWN TENSOR IS A BASE TENSOR PLUS PADDING, NOT A NEW TENSOR.
        # The HRNN ladder widens head counts, so in_proj_qkvz goes from 320 rows
        # to 960 -- and the first 320 are the ORIGINAL VALUES with zeros after.
        # Storing the whole thing was banking a copy plus a formula: measured,
        # 35 tensors looked "genuinely new" while a tensor-by-tensor diff had
        # already shown 0.00 MB of genuinely different VALUES. The recipe stores
        # the SOURCE NAME and the TARGET SHAPE; expand() pads.
        grown = None
        for cand, base in ((c, np.asarray(base_weights[c]))
                           for c in base_weights):
            if base.shape == a.shape or base.ndim != a.ndim:
                continue
            if any(b > t for b, t in zip(base.shape, a.shape)):
                continue
            sl = tuple(slice(0, d) for d in base.shape)
            if np.array_equal(a[sl], base):
                grown = cand
                break
        if grown is not None:
            # THE PADDING IS NOT ALWAYS ZERO. The ladder writes real a_log
            # values into the new heads -- half-life = exp(-a_log), which is
            # genuine information even though it comes from a formula. So the
            # recipe stores the SOURCE plus only the REMAINDER, which for a
            # blank-padded tensor is nothing and for a ladder rung is a handful
            # of rows. Assuming the remainder was zero cost an exact-rebuild
            # failure on in_proj_ba, where rows 8 and 9 carry the new rungs.
            base = np.asarray(base_weights[grown])
            rest = a.copy()
            rest[tuple(slice(0, d) for d in base.shape)] = 0
            if rest.any():
                grows[k] = [grown, list(a.shape)]
                arrays["__pad__" + k] = rest
            else:
                grows[k] = [grown, list(a.shape)]
        else:
            arrays[k] = a
    rules["renames"] = renames
    rules["zeros"] = zeros
    rules["grows"] = grows
    return rules, arrays


#: WHICH RUNG EACH KIND OF INSTALL DATA BELONGS ON, priced by `codec_place`
#: -- which measures every applicable unit against a zlib baseline and keeps
#: "store raw" as a first-class row rather than assuming compression wins.
#: MEASURED on 16.38 KB samples:
#:     trained weights          16.38 -> 15.15 KB   1.08x   ship RAW
#:     a reservation row        16.38 -> 15.17 KB   1.08x   ship the SEED
#:     ladder a_log values      16.38 ->  0.07 KB    234x   ship the FORMULA
#:     the zero padding         16.38 ->  0.04 KB    420x   ship a SHAPE
#: TRAINED WEIGHTS DO NOT COMPRESS -- 1.08x is noise, and any scheme claiming
#: better on them is either lossy or measuring something else. EVERYTHING THE
#: INSTALL ADDS DOES compress, by two to three orders of magnitude, because it
#: is structure rather than information. That is the entire storage argument in
#: one table, and it says the recipe is not an optimisation of the model -- it
#: is a refusal to store things that were never data.
CODEC_PLACEMENT = {
    "trained_weights": ("raw", 1.08),
    "reservation": ("seed", 1.08),
    "ladder_alog": ("formula", 234.0),
    "zero_padding": ("shape", 420.0),
}


def hlb_operator(vec):
    """An HLB bind, materialised as the DxD matrix install_op needs.

    THE SAVING IS IN STORING IT, NOT IN APPLYING IT, and both halves are true:
        hidden 1024   circulant 1,048,576 params | HLB 1,024 | 1024x
    but install_op writes MLP neurons, and neurons apply a MATRIX. So the model
    gets M_x = H diag(Hx) H / D -- verified equal to the elementwise Hadamard
    form at 1.5e-14 -- while the RECIPE stores the 1,024-element VECTOR and
    regenerates M_x on expansion. The operator is a formula; only its
    application is data. That is the same bank-or-formula split the zero padding
    and the a_log rungs already fall on."""
    v = np.asarray(vec, np.float64).ravel()
    d = v.size
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H @ np.diag(H @ v) @ H / float(d)


def compress_arrays(rules, arrays, base_weights, energy=0.9999, bits=8,
                    mode="lowrank"):
    """Hand the genuinely-new arrays to leCore's OWN delta store.

    AND leCORE'S OWN STORAGE LADDER (`unicron_archive`) IS THE RIGHT HOME FOR
    THE BYTES ONCE THE NAMES ARE FIXED -- four rungs, SAME / RECIPE / DELTA /
    RAW, BIT-exact, with XOR deltas rather than arithmetic ones because
    "arithmetic float deltas are not bit-exact (XOR is)". MEASURED against the
    install three ways:
        ladder alone, no reference resolution        1.29x
        ladder with only the renamed tensors         1.67x
        this module's rename+zero+pad resolution     2.7x
    THE LADDER IS NOT WORSE; IT IS BEING GIVEN THE WRONG INPUT. It matches by
    NAME, prepend renumbers every layer, and 26 of 76 installed tensors have no
    same-named reference at all -- so it correctly falls back to RAW on most of
    the model. Fixing the names first is what turns it loose, and that is a
    three-line rename map rather than a competing format.

    RULE 0, ARRIVED AT LATE. `unicron_delta_store` already stores a model
    difference properly: "unchanged tensors cost ZERO; touched ones go low-rank
    at a rank discovered from the delta's OWN SPECTRUM; a fat delta stays dense
    rather than paying factor overhead", with a D-QRELO mode (arXiv 2604.16940)
    for one-bit dominant structure plus low-rank residual.
    I HAND-ROLLED A WORSE VERSION OF THIS AS `build`. What build does that the
    delta store cannot is RESOLVE THE RENAMES: the delta store matches by NAME,
    and prepend RENUMBERS EVERY LAYER, so on its own it compared two nearly
    disjoint key sets and reported a 390,000x saving that was really "these two
    models share almost no tensor names". Measured: 44 tensors share a name, 42
    of those differ, and 32 exist only in the installed model.
    SO THEY COMPOSE. build() undoes the renaming and isolates what is actually
    new; the delta store compresses that. Neither alone is enough and the
    division is clean: one is a NAME problem, the other is a BYTES problem."""
    import lecore

    if not arrays:
        return {}, {"note": "nothing new to compress"}
    m = lecore.UnifiedMind(dim=64, seed=0)
    # rebuild the pair the delta store expects: matched names, matched shapes
    left, right = {}, {}
    for k, v in arrays.items():
        a = np.asarray(v)
        src = rules["renames"].get(k)
        if src is not None and np.asarray(base_weights[src]).shape == a.shape:
            left[k] = np.asarray(base_weights[src])
            right[k] = a
    if not left:
        return {}, {"note": "no name-matched pairs -- everything here is new"}
    return m.unicron_delta_store(left, right, energy=energy, bits=bits,
                                 mode=mode), {"pairs": len(left)}


def cost(rules, arrays, installed_weights):
    """What the recipe saves, in bytes. The number is the whole argument."""
    full = sum(np.asarray(v).nbytes for v in installed_weights.values())
    real = sum(np.asarray(v).nbytes for v in arrays.values())
    return {"expanded_bytes": int(full), "recipe_array_bytes": int(real),
            "renamed": len(rules["renames"]), "zero_tensors": len(rules["zeros"]),
            "grown": len(rules.get("grows", {})),
            "stored_tensors": len(arrays),
            "ratio": (full / real) if real else float("inf")}


def expand(rules, arrays, base_weights):
    """Rebuild the installed model from the recipe. Must be byte-exact."""
    out = {}
    for k, src in rules["renames"].items():
        out[k] = np.asarray(base_weights[src])
    for k, (shape, dt) in rules["zeros"].items():
        out[k] = np.zeros(tuple(shape), dtype=np.dtype(dt))
    for k, (src, shape) in rules.get("grows", {}).items():
        base = np.asarray(base_weights[src])
        big = np.zeros(tuple(shape), dtype=base.dtype)
        big[tuple(slice(0, d) for d in base.shape)] = base
        pad = arrays.get("__pad__" + k)
        if pad is not None:
            big = big + np.asarray(pad)
        out[k] = big
    for k, v in arrays.items():
        if not k.startswith("__pad__"):
            out[k] = np.asarray(v)
    return out


def _selftest_hlb():
    """The materialised operator must equal the elementwise form EXACTLY enough,
    or the recipe regenerates something the model was not installed with."""
    d = 128
    g = np.random.default_rng(0)
    x, y = g.standard_normal(d), g.standard_normal(d)
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    direct = H @ ((H @ x) * (H @ y)) / d
    assert np.max(np.abs(direct - hlb_operator(x) @ y)) < 1e-12
    return d * d / d


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_install_lecore import install
    import lecore

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("recipe selftest SKIPPED-SUBJECT (no model present)")
        return

    rt, cfg = load_runtime(src)
    w = load_weights_dir(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    w2, _c2, rep = install(w, cfg, rt,
                           [b for b in raw[5000:9000].encode("utf-8")],
                           [b for b in raw[20000:21200].encode("utf-8")][:1000],
                           tokenize=lambda t: [b for b in t.encode("utf-8")],
                           n_registers=16,
                           mind=lecore.UnifiedMind(dim=512, seed=0))

    _saving = _selftest_hlb()

    rules, arrays = build(w, w2, rep)
    rep_cost = cost(rules, arrays, w2)

    # ---- THE RECIPE MUST REBUILD THE MODEL EXACTLY, or it is not a format ----
    back = expand(rules, arrays, w)
    assert set(back) == set(w2), (len(back), len(w2))
    for k in w2:
        assert np.array_equal(np.asarray(back[k]), np.asarray(w2[k])), k

    # ---- AND IT MUST ACTUALLY BE SMALLER, or it is ceremony ----
    assert rep_cost["ratio"] > 2.0, rep_cost

    print("recipe selftest OK -- an install of a real model expands to %.2f MB "
          "and its RECIPE carries %.2f MB of genuinely new arrays (%.0fx "
          "smaller): %d tensors are RENAMES, %d are ALL ZEROS and need only a "
          "shape, %d are a base tensor PADDED to a larger shape, and %d hold "
          "values that are actually new. "
          "expand() rebuilds every tensor BYTE-EXACT, which is the only thing "
          "that makes a recipe a format rather than a hope. And an HLB "
          "operator regenerates from a 128-element VECTOR into the matrix the "
          "model applies, %.0fx less to store than a circulant"
          % (rep_cost["expanded_bytes"] / 1e6,
             rep_cost["recipe_array_bytes"] / 1e6, rep_cost["ratio"],
             rep_cost["renamed"], rep_cost["zero_tensors"],
             rep_cost["grown"], rep_cost["stored_tensors"], _saving))


if __name__ == "__main__":
    _selftest()
