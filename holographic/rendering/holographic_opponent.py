"""holographic_opponent.py -- cross-source disagreement as STRUCTURED computation (a faithful port of leOS).

WHY THIS EXISTS
---------------
When two sources encode the SAME thing -- two embedding models, two solvers, two forked worlds, two farm nodes --
their disagreement is not scalar noise. It's a structured vector with directional information. This decomposes that
disagreement into channels, by analogy to the human visual system's OPPONENT PROCESSING (red-green, blue-yellow,
luminance):

  agreement          -- what BOTH sources see (the shared ground truth), element-wise.
  a_exclusive        -- what A sees that B doesn't (A minus A's projection onto B's direction).
  b_exclusive        -- what B sees that A doesn't (B minus B's projection onto A's direction).
  magnitude_dispute  -- same direction, different confidence (only meaningful when roughly aligned).
  purple             -- a_exclusive + b_exclusive: the EMERGENT signal that exists in NEITHER source alone.
  divergence_score   -- scalar disagreement: the geodesic (angular) distance between the two directions.

The purple channel is the valuable one. Like the colour purple -- which has no wavelength and is invented by the
brain's opponent processing -- the purple signal is genuine computation the topology performs: it's non-zero exactly
because the two exclusives are measured against DIFFERENT references (A orthogonal-to-B, B orthogonal-to-A), so they
don't cancel. (For a=b it's zero -- nothing emergent; for a orthogonal to b it's a+b -- everything is emergent.)

COMPATIBILITY (kept exactly): the channel math here mirrors leOS's project/subsystems/opponent_channels.py so that
anything built on the leOS opponent channels behaves identically here. Field names, the agreement definition
(sign-match * min-magnitude), the cross-projection exclusives, magnitude_dispute, purple = a_exclusive + b_exclusive,
and the geodesic divergence are the leOS contract, not a re-derivation. leCore adds nothing that changes those numbers;
it only ports them to numpy/stdlib with the engine's own helpers.

This is a PAIRWISE (A vs B) operation, as in leOS -- the opponent metaphor is inherently two-sided. Consumers that
have several estimates (fork/merge with >2 forks, N-node voting) reduce pairwise over the shared consensus.

Numpy only; deterministic.
"""
import numpy as np


DEFAULT_INTERRUPT_THRESHOLD = 0.35        # radians of geodesic divergence above which the disagreement "interrupts"


def opponent_channels(vec_a, vec_b, interrupt_threshold=DEFAULT_INTERRUPT_THRESHOLD):
    """Decompose the disagreement between two vectors into the opponent channels (see the module docstring). Returns a
    dict of numpy-array channels plus the scalars: {agreement, a_exclusive, b_exclusive, magnitude_dispute, purple,
    divergence_score, cosine_similarity, interrupt, channel_magnitudes}. This is leOS's `decompose`, ported."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)

    if a.shape != b.shape:                                     # different-dim sources: compare on the shared prefix
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-10 or b_norm < 1e-10:                       # a zero vector has no direction -> empty channels
        return _empty_channels(len(a))
    a_hat = a / a_norm
    b_hat = b / b_norm

    # AGREEMENT: element-wise sign-agreement (sign(a)*sign(b): +1 same sign, -1 differ) times the SMALLER magnitude
    # -- the shared commitment per coordinate (positive where they agree in sign, negative where they oppose).
    agreement = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    # EXCLUSIVES: each source minus its projection onto the OTHER's direction -- the part only it sees. Note the two
    # projections use DIFFERENT references (A onto B, B onto A); that is what makes purple non-cancelling below.
    a_exclusive = a - np.dot(a, b_hat) * b_hat
    b_exclusive = b - np.dot(b, a_hat) * a_hat

    # MAGNITUDE DISPUTE: same direction, different strength -- only meaningful when the two roughly point the same way.
    cos_sim = float(np.dot(a_hat, b_hat))
    if cos_sim > 0.5:
        magnitude_dispute = (a_norm - b_norm) * (a_hat + b_hat) / 2.0
    else:
        magnitude_dispute = np.zeros_like(a)

    # PURPLE: the emergent signal present in neither source alone -- the sum of the two exclusives. (leOS contract.)
    purple = a_exclusive + b_exclusive

    # DIVERGENCE: the geodesic (angular) distance between the directions -- 0 identical, pi opposed.
    divergence = float(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    interrupt = bool(divergence > interrupt_threshold)

    channel_magnitudes = {
        "agreement": float(np.linalg.norm(agreement)),
        "a_exclusive": float(np.linalg.norm(a_exclusive)),
        "b_exclusive": float(np.linalg.norm(b_exclusive)),
        "magnitude_dispute": float(np.linalg.norm(magnitude_dispute)),
        "purple": float(np.linalg.norm(purple)),
    }
    return {
        "agreement": agreement,
        "a_exclusive": a_exclusive,
        "b_exclusive": b_exclusive,
        "magnitude_dispute": magnitude_dispute,
        "purple": purple,
        "divergence_score": divergence,
        "cosine_similarity": cos_sim,
        "interrupt": interrupt,
        "channel_magnitudes": channel_magnitudes,
    }


# leOS calls this `decompose`; keep that name too so ported code reads the same.
decompose = opponent_channels


def classify(channels, redundant_threshold=0.95, contradictory_threshold=0.3):
    """Classify the disagreement in a channel dict into one of leOS's five types: 'redundant' (they strongly agree),
    'contradictory' (they point very different ways), 'novel' (a big purple signal interrupted), 'hierarchical' (one
    source sees much more than the other), or 'complementary' (they see different aspects worth combining). Returns
    {type, confidence, description, divergence_score}."""
    mags = channels.get("channel_magnitudes", {})
    cos_sim = channels.get("cosine_similarity", 1.0)
    interrupt = channels.get("interrupt", False)
    a_excl = mags.get("a_exclusive", 0.0)
    b_excl = mags.get("b_exclusive", 0.0)
    purple_mag = mags.get("purple", 0.0)
    agree_mag = mags.get("agreement", 0.0)

    if cos_sim > redundant_threshold:
        dtype, confidence = "redundant", cos_sim
        desc = "sources strongly agree -- little new information from the comparison"
    elif cos_sim < contradictory_threshold:
        dtype, confidence = "contradictory", 1.0 - cos_sim
        desc = "sources point in very different directions -- fundamental disagreement"
    elif interrupt and purple_mag > agree_mag * 0.5:
        dtype = "novel"
        confidence = min(purple_mag / max(agree_mag, 0.01), 1.0)
        desc = "high purple signal -- emergent information present in neither source alone"
    elif a_excl > b_excl * 2 or b_excl > a_excl * 2:
        dtype = "hierarchical"
        confidence = max(a_excl, b_excl) / max(a_excl + b_excl, 0.01)
        desc = "one source sees significantly more than the other"
    else:
        dtype = "complementary"
        confidence = (a_excl + b_excl) / max(agree_mag + a_excl + b_excl, 0.01)
        desc = "sources see different aspects -- combine for a richer result"

    return {"type": dtype, "confidence": round(float(confidence), 3), "description": desc,
            "divergence_score": channels.get("divergence_score", 0.0)}


def blend(vec_a, vec_b, ratio=0.7):
    """Principled cross-source blend using the opponent structure (leOS's `blend`): keep the full agreement, mix the
    exclusives at `ratio` (default 70% A / 30% B), and add a small (0.1) purple contribution; renormalize. Returns the
    blended unit vector."""
    ch = opponent_channels(vec_a, vec_b)
    blended = (ch["agreement"] + ratio * ch["a_exclusive"] + (1.0 - ratio) * ch["b_exclusive"]
               + 0.1 * ch["purple"])
    n = np.linalg.norm(blended)
    return blended / n if n > 1e-10 else blended


def divergence(vec_a, vec_b):
    """Scalar disagreement: the geodesic (angular) distance between two vectors, in radians (0 = identical)."""
    return opponent_channels(vec_a, vec_b)["divergence_score"]


def agree(vec_a, vec_b, threshold=0.95):
    """Convenience gate: do the two sources agree? True iff cosine_similarity >= threshold (the abstention move --
    act when True, surface the conflict when False)."""
    return opponent_channels(vec_a, vec_b)["cosine_similarity"] >= threshold


def _empty_channels(dim):
    """Degenerate inputs (a zero vector): all-zero channels, zero divergence."""
    z = np.zeros(dim)
    return {"agreement": z, "a_exclusive": z, "b_exclusive": z, "magnitude_dispute": z, "purple": z,
            "divergence_score": 0.0, "cosine_similarity": 0.0, "interrupt": False,
            "channel_magnitudes": {"agreement": 0.0, "a_exclusive": 0.0, "b_exclusive": 0.0,
                                   "magnitude_dispute": 0.0, "purple": 0.0}}


def _selftest():
    rng = np.random.default_rng(0)
    dim = 512

    # --- identical sources: no exclusives, no purple, zero divergence, classified 'redundant' ---
    v = rng.standard_normal(dim)
    ch = opponent_channels(v, v)
    assert ch["divergence_score"] < 1e-6 and ch["cosine_similarity"] > 0.999
    assert ch["channel_magnitudes"]["purple"] < 1e-6, "identical -> nothing emergent"
    assert classify(ch)["type"] == "redundant"

    # --- orthogonal sources: purple = a+b (present!), high divergence ---
    a = np.zeros(dim); a[0] = 1.0
    b = np.zeros(dim); b[1] = 1.0
    ch2 = opponent_channels(a, b)
    assert np.allclose(ch2["purple"], a + b), "purple must be a_exclusive + b_exclusive (leOS)"
    assert ch2["channel_magnitudes"]["purple"] > 1.0
    assert ch2["divergence_score"] > 1.5                        # ~pi/2

    # --- the purple identity holds in general: purple == a_exclusive + b_exclusive ---
    x = rng.standard_normal(dim); y = rng.standard_normal(dim)
    ch3 = opponent_channels(x, y)
    assert np.allclose(ch3["purple"], ch3["a_exclusive"] + ch3["b_exclusive"])

    # --- agreement keeps sign-matched shared magnitude; opposed/zero coords cancel ---
    p = np.array([2.0, 3.0, -1.0, 0.0])
    q = np.array([1.0, 5.0,  4.0, 2.0])
    chg = opponent_channels(p, q)
    assert np.allclose(chg["agreement"], [1.0, 3.0, -1.0, 0.0])   # sign(a)*sign(b)*min|.| : coord2 signs differ -> -1

    # --- blend keeps agreement + weighted exclusives + a little purple, unit length ---
    bl = blend(x, y, ratio=0.7)
    assert abs(np.linalg.norm(bl) - 1.0) < 1e-6

    # --- convenience: agree()/divergence() ---
    assert agree(v, v) and not agree(a, b)
    assert divergence(v, v) < 1e-6

    print("OK: holographic_opponent self-test passed (leOS-faithful: identical -> redundant, no purple; orthogonal "
          "-> purple == a+b (emergent); purple == a_exclusive + b_exclusive always; agreement = sign-match * "
          "min-magnitude; blend is unit-length; agree()/divergence() convenience)")


if __name__ == "__main__":
    _selftest()
