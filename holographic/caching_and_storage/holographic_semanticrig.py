"""Semantic rig -- bones, hinges, and IK handles for the memory itself.

WHY THIS MODULE EXISTS (Moose's framing, taken literally): the 3D animation stack -- bones,
joints with constraints, skinning, FABRIK/CCD -- is constrained, weighted transform
propagation through a structure. leCore's memory IS such a structure, and the shufflebrain
session (docs/PANEL_pietsch_hologramic.md) measured exactly which transforms each substrate
survives coherently. So the framework can be RIGGED like a bound mesh: pull a handle (a
cue -> target contract) and the whole stored structure changes shape predictably, within
joint limits, losslessly.

THE SYMMETRY GROUPS PICK THE BONES (the shufflebrain finding made load-bearing):
  - GDN matrix memory carries the FULL orthogonal group -> bones are hinge-limited GIVENS
    PLANES. Disjoint planes COMMUTE, so CCD's closed-form per-joint angle (atan2) is exact,
    not approximate -- the solver recovered a planted pose to 8.3e-17 rad in the pilot.
  - HRR traces carry only the CYCLIC group -> bones are rfft PHASE BANDS (Puckette's phase
    vocoder as a skeleton). Per-band closed form phi = arg(sum conj(Z_tgt) Z_cur); planted
    phases recovered to 3.8e-07 rad, handle cos 1.000000.

THE POSE IS A NEW EDIT PRIMITIVE, priced differently from writing: a pose is an ISOMETRY --
recall fidelity is EXACTLY preserved (0.929 -> 0.929 measured), the inverse pose restores the
memory to machine precision (2.8e-17 matrix / 2.8e-13 trace), and bystander memories move
only within the touched planes (min self-cos 0.996 with 8 planes of 128 dims). Contrast
external_write (Ouroboros), which is ADDITIVE and pays crosstalk. Write when you need new
content; POSE when you need the same content in a new shape.

KEPT NEGATIVES (each measured, each pinned):
  - VALUE-SIDE POSE DIRECTION: S @ R gives readouts R^T w -- the INVERSE pose. The correct
    value-side pose is S @ R.T. The first pilot predicted co-articulation with the wrong
    direction and read 2.3e-01 where the theorem says 1e-16; the direction is now in the API,
    not the caller's head.
  - THE NYQUIST BIN IS REAL: an rfft phase bone that touches DC or Nyquist silently truncates
    the imaginary part at irfft -- the pose stops being unitary (restore degraded to 6.8e-05).
    Bands here exclude both by construction; the selftest pins restoration at machine scale.
  - THE ORBIT IS SMALL AND THE RIG SAYS SO: a far target floors honestly with hinges at their
    limits (7/8 slammed, cos 0.032 -> 0.055 in the pilot). A rig is not a rewrite; reach is
    bounded by joint count x limits, and the residual is the constraint telling the truth.

Delegations: bind/unbind from holographic_ai; the mesh-space IK/skin stack (solve_ik,
solve_ik_limited, skin_mesh) remains the geometric family this module is the semantic lift of.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class GivensRig:
    """Bones = disjoint 2-plane Givens hinges (commuting -> exact closed-form CCD) with angle
    limits, acting on a dk-dimensional value space. Pose vectors, solve handles, pose a whole
    GDN matrix memory value-side (S -> S @ R.T so readouts transform FORWARD -- the direction
    kept-negative lives in this line)."""

    def __init__(self, dk, n_bones=8, limit_deg=35.0, seed=0, planes=None):
        rng = np.random.default_rng(seed)
        self.dk = int(dk)
        # planes may be supplied explicitly -- including CHAINS with SHARED axes in the
        # Denavit-Hartenberg spirit (1955: a serial mechanism is an ordered product of
        # per-joint matrices). Shared axes make the sequence NON-commuting: CCD still reaches
        # the handle, but the pose that reaches it is no longer unique -- KINEMATIC REDUNDANCY,
        # the elbow-up/elbow-down of memory space (measured: handle 0.999991 with thetas
        # 2.2e-01 rad from the planted pose). A feature with a classical name, pinned as such.
        if planes is not None:
            self.planes = np.asarray(planes, int)
        else:
            self.planes = rng.choice(self.dk, size=(int(n_bones), 2), replace=False)
        self.limit = float(np.deg2rad(limit_deg))

    def pose_vec(self, vec, thetas, inverse=False):
        v = np.array(vec, float)
        for (i, j), t in zip(self.planes, -np.asarray(thetas) if inverse else np.asarray(thetas)):
            c, s = np.cos(t), np.sin(t)
            vi, vj = v[i], v[j]
            v[i], v[j] = c * vi - s * vj, s * vi + c * vj
        return v

    def rotation(self, thetas):
        """The pose as an explicit orthogonal matrix (columns = posed basis vectors)."""
        return np.stack([self.pose_vec(e, thetas) for e in np.eye(self.dk)], axis=1)

    def solve(self, current, target, sweeps=6):
        """CCD with the closed-form optimal angle per hinge, clamped to the limit. Disjoint
        planes commute, so per-joint atan2 is globally exact rather than locally greedy."""
        thetas = np.zeros(len(self.planes))
        tgt = np.asarray(target, float)
        for _ in range(int(sweeps)):
            for b, (i, j) in enumerate(self.planes):
                tb = thetas.copy()
                tb[b] = 0.0
                base = self.pose_vec(current, tb)
                A = tgt[i] * base[i] + tgt[j] * base[j]
                B = -tgt[i] * base[j] + tgt[j] * base[i]
                thetas[b] = float(np.clip(np.arctan2(B, A), -self.limit, self.limit))
        return thetas, self.pose_vec(current, thetas)

    def pose_memory(self, S, thetas):
        """Value-side pose of a GDN matrix memory: readouts transform FORWARD, exactly:
        (S @ R.T)^T k == pose_vec(S^T k). An isometry -- recall fidelity and capacity are
        untouched, and pose_memory(S, -thetas)... use the returned R for exact inversion."""
        R = self.rotation(thetas)
        return S @ R.T


class BandPhaseRig:
    """HRR-native bones: contiguous rfft bands each carrying one phase hinge. DC and Nyquist
    are EXCLUDED by construction (they must stay real -- the truncation kept-negative). The
    pose commutes with the HRR readout, so every stored value co-articulates exactly."""

    def __init__(self, dim, n_bones=8, limit_deg=60.0):
        self.dim = int(dim)
        self.edges = np.linspace(1, self.dim // 2, int(n_bones) + 1).astype(int)
        self.n = int(n_bones)
        self.limit = float(np.deg2rad(limit_deg))

    def pose(self, x, phis, inverse=False):
        X = np.fft.rfft(np.asarray(x, float))
        for b in range(self.n):
            X[self.edges[b]:self.edges[b + 1]] *= np.exp((-1j if inverse else 1j) * float(phis[b]))
        return np.fft.irfft(X, n=self.dim)

    def solve(self, current, target, sweeps=4):
        """Per-band closed form: phi_b = arg(sum_band conj(Z_target) Z_current), clamped."""
        phis = np.zeros(self.n)
        Wt = np.fft.rfft(np.asarray(target, float))
        for _ in range(int(sweeps)):
            for b in range(self.n):
                pb = phis.copy()
                pb[b] = 0.0
                Zb = np.fft.rfft(self.pose(current, pb))
                sl = slice(self.edges[b], self.edges[b + 1])
                z = np.sum(np.conj(Zb[sl]) * Wt[sl])
                phis[b] = float(np.clip(np.angle(z), -self.limit, self.limit))
        return phis, self.pose(current, phis)


def data_aligned_planes(S, n_bones=8):
    """R4 -- rig-from-parts for memory: bones from THE DATA'S OWN JOINTS. SVD the memory and
    take consecutive right-singular pairs as planes in the singular basis. MEASURED: 2x the
    handle reach of random planes at the same joint budget (0.112 vs 0.058, 8 joints) --
    the holographic framework rigging itself is not decoration, it is reach per joint.
    Returns (planes, basis); solve in the basis, pose with basis @ R @ basis.T.

    RANK CAP (a fresh-seed clean-extract taught this): a memory of rank r has only r live
    singular directions -- planes beyond r/2 are NULLSPACE JOINTS, hinges welded to nothing,
    and a rig that spends its budget there loses to random placement. Bones are capped at
    the effective rank; ask for more and you get what the data can actually articulate."""
    U, sg, Vt = np.linalg.svd(np.asarray(S, float))
    r_eff = int(np.sum(sg > sg[0] * 1e-9))
    nb = min(int(n_bones), max(r_eff // 2, 1))
    planes = [(2 * i, 2 * i + 1) for i in range(nb)]
    return planes, Vt.T


def key_pose(S, rig, thetas):
    """R5 -- pose the KEY side: S' = R S. Content at MOVED addresses is EXACT
    ((RS)^T (Rk) = S^T k, measured 6e-17): the memory is re-ADDRESSED, not re-written.
    Old addresses drift by exactly the key's mass in the posed planes -- small rigs barely
    move them, full-coverage rigs retire them. The third mouth verb: WRITE adds content,
    POSE reshapes values, KEY-POSE relocates addresses."""
    R = rig.rotation(thetas)
    return R @ np.asarray(S, float)


def twist_split_norms(dim=128, seed=0):
    """The DQS-era production fix, lifted and priced: riggers defeat the candy-wrapper by
    ADDING TWIST BONES -- routing a big blend through half-angle intermediates. Measured law:
    one 90-degree blend collapses to cos(45)=0.707; via a 45-degree intermediate it holds
    cos(22.5)=0.924. Two small blends beat one big one, by exactly the half-angle cosine."""
    rng = np.random.default_rng(seed)
    v = np.random.default_rng(seed).standard_normal(dim)
    v = v / np.linalg.norm(v)
    pl = [(i, dim // 2 + i) for i in range(dim // 2)]
    def pose(x, ang):
        x = np.array(x, float)
        c, s = np.cos(ang), np.sin(ang)
        for (i, j) in pl:
            xi, xj = x[i], x[j]
            x[i], x[j] = c * xi - s * xj, s * xi + c * xj
        return x
    one = float(np.linalg.norm(0.5 * pose(v, np.deg2rad(90)) + 0.5 * v))
    h = pose(v, np.deg2rad(45))
    two = float(np.linalg.norm(0.5 * pose(h, np.deg2rad(45)) + 0.5 * h))
    return {"one_stage": one, "two_stage": two}


class SkinnedRig:
    """R1 -- skinning weights proper: partition KEY space into bone regions and pose each
    region with its own transform, S' = sum_b P_b S R_b^T. A key inside region b reads its
    region's pose EXACTLY; a straddling key reads the weighted blend -- linear blend skinning
    with the key as the vertex.

    DESIGN-FOR-RIGGING (the mesh lore, lifted): good rigging requires good topology. With
    ORTHOGONAL key regions every contract is machine-exact (posed 1e-16, untouched 1e-16,
    blend exact). With RANDOM keys the regions overlap and isolation leaks at a PRICED scale,
    mean ||P_A k_B|| ~ sqrt(n_A/D) (measured 0.304 vs 0.250 at 8/128) -- leak() reports it
    rather than hiding it, because a rig that lies about isolation is worse than no rig.

    KEPT NEGATIVE, with its mesh name: THE CANDY-WRAPPER. Linearly blending a large rotation
    with identity shrinks the readout norm, exactly as LBS collapses a twisted joint --
    measured 0.974 / 0.836 / 0.707 at 8/32/64-of-64 planes covered, hitting cos(45 deg)
    EXACTLY at full coverage. Severity = rotated-mass fraction x (1 - cos): the mesh
    artifact, now with a quantitative law. The mesh world's fix (dual quaternions / slerp)
    has no one-matrix analog here -- per-key slerp is not a linear memory edit -- so the
    artifact is priced and pinned instead of patched."""

    def __init__(self, key_basis_a, dk):
        B = np.linalg.qr(np.stack(key_basis_a, axis=1))[0]
        self.PA = B @ B.T
        self.PB = np.eye(int(dk)) - self.PA

    def leak(self, other_keys):
        """Mean ||P_A k|| over keys meant to be OUTSIDE region A -- the isolation price."""
        return float(np.mean([np.linalg.norm(self.PA @ np.asarray(k, float))
                              for k in other_keys]))

    def pose_memory(self, S, rig_a, thetas_a):
        """Region A gets rig_a's pose; region B stays put. Extend with more (P_b, R_b) terms
        for more bones -- the partition-of-unity structure is the LBS contract."""
        R = rig_a.rotation(thetas_a)
        return self.PA @ S @ R.T + self.PB @ S


def semantic_rig_battery(dim=128, hrr_dim=2048, n_items=20, seed=0):
    """The pilot as a repeatable battery: both substrates, all six contracts -- reachable-handle
    recovery, far-handle honesty, exact co-articulation, isolation, losslessness, inverse
    restoration. Deterministic in (dim, hrr_dim, n_items, seed)."""
    rng = np.random.default_rng(seed)
    # -- GDN side --
    Ks = [_unit(rng.standard_normal(dim)) for _ in range(n_items)]
    Vs = [_unit(rng.standard_normal(dim)) for _ in range(n_items)]
    S = np.zeros((dim, dim))
    for k, v in zip(Ks, Vs):
        S = 0.98 * S + np.outer(k, v)
    rig = GivensRig(dim, seed=seed)
    w0 = S.T @ Ks[0]
    t_plant = rng.uniform(-0.5, 0.5, len(rig.planes)) * rig.limit
    tgt = _unit(rig.pose_vec(w0, t_plant))
    th, w1 = rig.solve(w0, tgt)
    Sp = rig.pose_memory(S, th)
    coart = max(float(np.max(np.abs(Sp.T @ k - rig.pose_vec(S.T @ k, th)))) for k in Ks)
    bys = min(float(_unit(Sp.T @ k) @ _unit(S.T @ k)) for k in Ks[1:])
    rec0 = float(np.mean([float(_unit(S.T @ k) @ v) for k, v in zip(Ks, Vs)]))
    rec1 = float(np.mean([float(_unit(Sp.T @ k) @ _unit(rig.pose_vec(v, th))) for k, v in zip(Ks, Vs)]))
    restore = float(np.max(np.abs(Sp @ rig.rotation(th) - S)))
    far = _unit(w0 + 1.2 * rng.standard_normal(dim))
    thf, wf = rig.solve(w0, far)
    gdn = {"planted_recovery_rad": float(np.max(np.abs(th - t_plant))),
           "handle_cos": float(_unit(w1) @ tgt),
           "far_handle_cos": float(_unit(wf) @ far),
           "hinges_at_limit_far": int(np.sum(np.isclose(np.abs(thf), rig.limit))),
           "coarticulation_err": coart, "bystander_min_cos": bys,
           "recall_before": rec0, "recall_after": rec1, "restore_err": restore}
    # -- HRR side --
    rngh = np.random.default_rng(seed + 1)
    hk = [_unit(rngh.standard_normal(hrr_dim)) for _ in range(n_items)]
    hv = [_unit(rngh.standard_normal(hrr_dim)) for _ in range(n_items)]
    T = np.sum([bind(k, v) for k, v in zip(hk, hv)], axis=0)
    brig = BandPhaseRig(hrr_dim)
    phis_p = rngh.uniform(-0.6, 0.6, brig.n) * brig.limit
    Tp = brig.pose(T, phis_p)
    hcoart = max(float(np.max(np.abs(unbind(Tp, k) - brig.pose(unbind(T, k), phis_p)))) for k in hk)
    wh = unbind(T, hk[0])
    ph, wh1 = brig.solve(wh, brig.pose(wh, phis_p))
    hrestore = float(np.max(np.abs(brig.pose(Tp, phis_p, inverse=True) - T)))
    hrr = {"coarticulation_err": hcoart,
           "planted_recovery_rad": float(np.max(np.abs(ph - phis_p))),
           "handle_cos": float(_unit(wh1) @ _unit(brig.pose(wh, phis_p))),
           "restore_err": hrestore}
    # -- skinned lane (R1): ortho regions exact, leak law priced, candy-wrapper pinned --
    rngs = np.random.default_rng(seed + 2)
    Q = np.linalg.qr(rngs.standard_normal((dim, 16)))[0]
    KA, KB = [Q[:, i] for i in range(8)], [Q[:, 8 + i] for i in range(8)]
    VA = [_unit(rngs.standard_normal(dim)) for _ in range(8)]
    VB = [_unit(rngs.standard_normal(dim)) for _ in range(8)]
    S2 = np.zeros((dim, dim))
    for k, v in list(zip(KA, VA)) + list(zip(KB, VB)):
        S2 = 0.98 * S2 + np.outer(k, v)
    srig = SkinnedRig(KA, dim)
    rig2 = GivensRig(dim, seed=seed + 3)
    th2 = rngs.uniform(-0.5, 0.5, len(rig2.planes)) * rig2.limit
    Sp2 = srig.pose_memory(S2, rig2, th2)
    e_a = max(float(np.max(np.abs(Sp2.T @ k - rig2.pose_vec(S2.T @ k, th2)))) for k in KA)
    e_b = max(float(np.max(np.abs(Sp2.T @ k - S2.T @ k))) for k in KB)
    w = 0.5
    kmix = w * KA[0] + np.sqrt(1 - w * w) * KB[0]
    blend_err = float(np.max(np.abs(Sp2.T @ kmix - (w * rig2.pose_vec(S2.T @ KA[0], th2)
                                                    + np.sqrt(1 - w * w) * (S2.T @ KB[0])))))
    leak = SkinnedRig([_unit(rngs.standard_normal(dim)) for _ in range(8)], dim).leak(
        [_unit(rngs.standard_normal(dim)) for _ in range(8)])
    v = _unit(rngs.standard_normal(dim))
    full = GivensRig(dim, n_bones=dim // 2, limit_deg=90.0, seed=seed + 4)
    cw = float(np.linalg.norm(0.5 * full.pose_vec(v, np.full(dim // 2, np.deg2rad(90))) + 0.5 * v))
    skinned = {"region_a_posed_err": e_a, "region_b_untouched_err": e_b,
               "lbs_blend_err": blend_err, "leak_random_regions": leak,
               "leak_law_sqrt_na_over_d": float(np.sqrt(8 / dim)),
               "candy_wrapper_full_coverage": cw}
    # -- R2 chain lane: shared axes, redundancy pinned as a FINDING --
    rngc = np.random.default_rng(seed + 5)
    chain = GivensRig(dim, limit_deg=35.0, planes=[(i, i + 1) for i in range(6)])
    wch = _unit(rngc.standard_normal(dim))
    th_p = rngc.uniform(-0.6, 0.6, 6) * chain.limit
    tgt_ch = chain.pose_vec(wch, th_p)
    th_c, w_c = chain.solve(wch, tgt_ch, sweeps=30)
    chain_res = {"handle_cos": float(_unit(w_c) @ _unit(tgt_ch)),
                 "theta_err": float(np.max(np.abs(th_c - th_p)))}
    # -- R4 data-aligned bones vs random, same budget --
    # PAIRED instrument (the clean-extract at a fresh seed caught the one-draw version losing
    # its margin to a lucky random rig): SAME three targets for both arms, reach = the mean.
    pl_d, basis = data_aligned_planes(S, 8)
    rig_d = GivensRig(dim, planes=pl_d)
    rig_r = GivensRig(dim, n_bones=len(pl_d), seed=seed + 7)   # SAME budget: fair pairing
    rngt = np.random.default_rng(seed + 6)
    reach_d = reach_r = 0.0
    for _ in range(3):
        tgt_far = _unit(w0 + 1.0 * rngt.standard_normal(dim))
        thd, wd = rig_d.solve(basis.T @ w0, basis.T @ tgt_far, sweeps=8)
        reach_d += float(_unit(basis @ wd) @ tgt_far) / 3.0
        thr, wr = rig_r.solve(w0, tgt_far, sweeps=8)
        reach_r += float(_unit(wr) @ tgt_far) / 3.0
    # -- R5 key-side pose --
    kp = GivensRig(dim, seed=seed + 8)
    th_k = np.random.default_rng(seed + 8).uniform(-0.5, 0.5, len(kp.planes)) * kp.limit
    Sk = key_pose(S, kp, th_k)
    Rk = kp.rotation(th_k)
    readdr = max(float(np.max(np.abs(Sk.T @ (Rk @ np.asarray(k)) - S.T @ k))) for k in Ks)
    ts = twist_split_norms(dim, seed)
    return {"gdn": gdn, "hrr": hrr, "skinned": skinned,
            "chain": chain_res,
            "data_aligned": {"reach_data": reach_d, "reach_random": reach_r},
            "key_pose_readdress_err": readdr,
            "twist": ts}


def _selftest():
    # PLANTED TRUTHS from the pilot; each pin is one of the six rig contracts. Kept negatives
    # (pose direction, Nyquist truncation, orbit honesty) are structural: the direction lives
    # in pose_memory's one line, the band edges exclude DC/Nyquist by construction, and the
    # far-handle pin asserts hinges AT their limits rather than pretending reach.
    r = semantic_rig_battery(dim=128, hrr_dim=2048, n_items=20, seed=0)
    g = r["gdn"]
    assert g["planted_recovery_rad"] < 1e-12, g            # exact CCD (commuting hinges)
    assert g["handle_cos"] > 0.999999, g
    assert g["coarticulation_err"] < 1e-12, g              # the direction kept-negative, pinned
    assert g["bystander_min_cos"] > 0.98, g                # isolation
    assert abs(g["recall_before"] - g["recall_after"]) < 1e-9, g   # pose is an isometry
    assert g["restore_err"] < 1e-12, g                     # exactly invertible
    assert g["far_handle_cos"] < 0.3 and g["hinges_at_limit_far"] >= 5, g  # orbit honesty
    h = r["hrr"]
    assert h["coarticulation_err"] < 1e-12, h              # readout commutes with band pose
    assert h["planted_recovery_rad"] < 1e-5, h
    assert h["handle_cos"] > 0.999999, h
    assert h["restore_err"] < 1e-9, h                      # Nyquist excluded -> unitary
    s = r["skinned"]
    assert s["region_a_posed_err"] < 1e-12 and s["region_b_untouched_err"] < 1e-12, s
    assert s["lbs_blend_err"] < 1e-10, s                   # LBS contract exact under ortho topology
    assert 0.5 * s["leak_law_sqrt_na_over_d"] < s["leak_random_regions"] < 3 * s["leak_law_sqrt_na_over_d"], s
    assert abs(s["candy_wrapper_full_coverage"] - np.cos(np.deg2rad(45))) < 1e-3, s  # the mesh artifact, quantitatively
    c = r["chain"]
    assert c["handle_cos"] > 0.999, c                      # CCD reaches through a non-commuting chain
    assert c["theta_err"] > 1e-3, c                        # ...but the pose is NOT unique: kinematic
                                                           # redundancy, pinned as the finding it is
    d = r["data_aligned"]
    assert d["reach_data"] > 1.3 * d["reach_random"], d    # the data's own joints out-reach random
    assert r["key_pose_readdress_err"] < 1e-12, r["key_pose_readdress_err"]  # re-addressed, exact
    t = r["twist"]
    assert abs(t["one_stage"] - np.cos(np.deg2rad(45))) < 1e-3, t
    assert abs(t["two_stage"] - np.cos(np.deg2rad(22.5))) < 1e-3, t  # the rigger's fix, by the half-angle law
    print("OK: semantic rig -- Givens hinges recover a planted pose to %.0e rad and pose the "
          "matrix memory losslessly (recall %.3f == %.3f, restore %.0e); band-phase bones "
          "co-articulate the HRR trace to %.0e; far handles floor honestly at the joint limits"
          % (g["planted_recovery_rad"], g["recall_before"], g["recall_after"],
             g["restore_err"], h["coarticulation_err"]))
    print("    skinned: ortho regions exact (%.0e/%.0e), leak law priced (%.3f ~ %.3f), "
          "candy-wrapper at full coverage %.3f == cos45" % (s["region_a_posed_err"],
          s["region_b_untouched_err"], s["leak_random_regions"],
          s["leak_law_sqrt_na_over_d"], s["candy_wrapper_full_coverage"]))


if __name__ == "__main__":
    _selftest()
