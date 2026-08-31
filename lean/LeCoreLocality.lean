/-
L1: LOCALITY OF BLENDSHAPE CORRECTIVES -- PROVED, not sampled.

WHY. SMPL's pose correctives are dense: they "relate every vertex on the mesh to all the
joints in the kinematic tree, capturing spurious long-range correlations" -- artifacts STAR
(Osman et al. 2020) calls "unappealing for animators". STAR removes them by TRAINING FROM
SCANS to learn each joint's activation region. leCore's holographic_blendbasis DECLARES the
region instead and MEASURED zero overreach; but a measurement over one mesh and a few
correctives is evidence, not a guarantee. These theorems close that gap for ALL meshes, ALL
sources, ALL radii and ALL amplitudes at once.

THE DOMAIN, and why it is not the reals. Our Lean is Tier-1 bare (no mathlib), so there is no
ℝ, no LinearOrder class, not even max_eq_right or zero_mul -- all of that lives in mathlib.
Rather than declare the proof impossible, LEVER 4: lift the statement to a domain where it IS
decidable. Fixed-point Int at 1/1000 is that domain, `omega` discharges linear integer
arithmetic in core Lean, and the lift is honest in both directions -- a GPU evaluating this
in fixed point would compute exactly these values, and the support property is a fact about
the CLIP, which is order-theoretic and survives any faithful numeric domain.

WHAT IS PROVED, mirroring holographic_blendbasis.support_weights:
  clip_zero              the clip really does floor at zero
  clip_bounds            weights never leave [0, 1]  (never invert, never overshoot)
  ramp_zero_outside      d >= r  =>  the ramp is EXACTLY zero, not merely small
  weight_zero_outside    d >= r  =>  the smoothstep weight is EXACTLY zero  <- L1 itself
  disp_zero_outside      => the DISPLACEMENT is zero for EVERY amplitude, including the
                            extrapolated ones animators actually use
  sum_zero_outside       => a STACK of correctives is zero where each is out of support,
                            which is the composition statement one corrective cannot give

SCOPE, kept honest. This proves the SUPPORT property: declared radii are respected exactly,
so no corrective can reach a vertex outside its stated region. It does NOT claim a given
radius is anatomically right -- that is a modelling choice no proof supplies -- and it takes
the geodesic distance as given, since proving Dijkstra-over-edges approximates the true
surface geodesic is a separate obligation, not claimed here.
-/

namespace LeCore

/-- Fixed-point scale: 1000 represents 1.0. -/
def SCALE : Int := 1000

/-- Clip to [0, 1] in fixed point -- the operation that makes support EXACT rather than
asymptotic. -/
def clip (x : Int) : Int := if x ≤ 0 then 0 else if x ≥ SCALE then SCALE else x

/-- The clip floors at zero: anything at or below zero is zero, exactly. -/
theorem clip_zero (x : Int) (h : x ≤ 0) : clip x = 0 := by
  unfold clip; simp [h]

/-- A weight never leaves [0,1], so a corrective can neither invert its displacement nor
overshoot the amplitude it was handed. -/
theorem clip_bounds (x : Int) : 0 ≤ clip x ∧ clip x ≤ SCALE := by
  unfold clip SCALE
  split
  · omega
  · split <;> omega

/-- The linear ramp: 1 at the anchor, falling to 0 at the declared radius `r`. -/
def ramp (d r : Int) : Int := clip (SCALE - SCALE * d / r)

/-- **Support is exactly bounded.** At or beyond the declared radius the ramp is ZERO --
not small, not asymptotically negligible. This is what makes "declared locality" a
guarantee rather than a hope. -/
theorem ramp_zero_outside (d r : Int) (hr : 0 < r) (h : r ≤ d) : ramp d r = 0 := by
  unfold ramp SCALE
  apply clip_zero
  have hd : 1000 * r ≤ 1000 * d := by omega
  have : (1000 : Int) ≤ 1000 * d / r := Int.le_ediv_iff_mul_le hr |>.mpr (by omega)
  omega

/-- Smoothstep u^2 (3 - 2u) in fixed point -- C1 at both ends, where a linear ramp would
leave a visible crease exactly at the support boundary. -/
def smoothstep (u : Int) : Int := u * u * (3 * SCALE - 2 * u) / (SCALE * SCALE)

theorem smoothstep_zero : smoothstep 0 = 0 := by decide

/-- **L1 -- THE LOCALITY THEOREM.** A corrective's weight is exactly zero at and beyond its
declared radius, for every distance and every radius. This is the property STAR needed a
scan dataset to obtain, holding here by construction and now by proof. -/
theorem weight_zero_outside (d r : Int) (hr : 0 < r) (h : r ≤ d) :
    smoothstep (ramp d r) = 0 := by
  rw [ramp_zero_outside d r hr h]
  exact smoothstep_zero

/-- The displacement one corrective applies: amplitude times weight. -/
def disp (amp d r : Int) : Int := amp * smoothstep (ramp d r)

/-- **Locality survives ANY amplitude**, including the extrapolated values outside [0,1] that
animators drive blendshapes with. A bound that failed under extrapolation would be useless in
production. -/
theorem disp_zero_outside (amp d r : Int) (hr : 0 < r) (h : r ≤ d) : disp amp d r = 0 := by
  unfold disp
  rw [weight_zero_outside d r hr h]
  omega

/-- A stack of correctives, summed. -/
def stack : List (Int × Int × Int) → Int
  | [] => 0
  | (amp, d, r) :: rest => disp amp d r + stack rest

/-- **COMPOSITION.** A whole stack of correctives is zero wherever every one of them is out
of its own support. This is the statement a single-corrective proof does NOT give, and the
one a rig actually needs: a face carries dozens of correctives and the guarantee must hold
for the entire stack, not one at a time. -/
theorem stack_zero_outside :
    ∀ (cs : List (Int × Int × Int)),
      (∀ c ∈ cs, 0 < c.2.2 ∧ c.2.2 ≤ c.2.1) → stack cs = 0
  | [], _ => by unfold stack; rfl
  | (amp, d, r) :: rest, h => by
      have hhead : 0 < r ∧ r ≤ d := h (amp, d, r) (List.mem_cons_self _ _)
      have hrest : ∀ c ∈ rest, 0 < c.2.2 ∧ c.2.2 ≤ c.2.1 :=
        fun c hc => h c (List.mem_cons_of_mem _ hc)
      unfold stack
      rw [disp_zero_outside amp d r hhead.1 hhead.2, stack_zero_outside rest hrest]
      rfl

end LeCore
