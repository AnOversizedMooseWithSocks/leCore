/-
HEAD SPEC INVARIANTS: every parameter vector in range is a HEAD.

WHY THIS IS THE RIGHT THEOREM. Three separate fitting formulations converged cleanly and
produced meaningless geometry: 9 capsules scoring 3.34x baseline that looked like scattered
blobs, and a 44%-better fit that was a flat PANCAKE. Each time the objective had a null space
and the optimiser found it, exactly as an optimiser should.

Proving a given objective identifiable is hard and must be redone for every objective.
CONSTRAINING THE PARAMETERISATION so every point in it is anatomically well-formed is
tractable AND STRICTLY STRONGER: a pancake stops being a reachable solution at all, so NO
objective -- however badly designed -- can return one, and the guarantee transfers to every
future fit for free.

These theorems prove exactly what holographic_headspec.check_invariants tests, but for ALL
parameters in the admissible box rather than the 400 random vectors the selftest draws.

TWO REFORMULATIONS WERE NEEDED TO MAKE `omega` DECIDE THIS, and both are worth recording
because they are the standard moves:
  * The width/height coupling was first written as a PRODUCT (skullH = skullHF * skullW /
    10000). omega is linear-only and treats a product of two variables as opaque, so the
    aspect proof was unreachable. Stating the coupling as LINEAR INEQUALITIES in `inRange`
    is equivalent and decidable.
  * Positions were first written with truncating integer division (34 * faceH / 100), then
    as direct fractions -- both still left omega chasing floors through a chain of divisions.
    THE FIX IS TO REMOVE DIVISION ENTIRELY: every position below is expressed at 100x SCALE,
    so the coefficients are exact integers and every statement is pure linear arithmetic.
    Scaling the output units instead of dividing is the right move whenever a proof only
    cares about ORDER, which is precisely what these invariants are about.

Fixed point at 1/10000, as in LeCoreLocality.lean: bare Lean has no reals, and every claim
here is an ordering fact about sums of positive quantities, which survives any faithful
numeric domain.
-/

namespace LeCore.HeadSpec

/-- Head parameters in fixed point. Heights are absolute; their coupling to the width is
stated as LINEAR constraints in `inRange`. -/
structure Params where
  skullW   : Int        -- half-width of the cranium
  skullH   : Int        -- crown height above the eye line
  faceH    : Int        -- eye line down to chin
  noseProj : Int        -- nose tip ahead of the eye line
  browZ    : Int        -- brow overhang, forward
  chinZ    : Int        -- chin projection, forward

/-- The admissible box, mirroring PARAM_RANGE. Clauses 5-8 TIE the vertical extents to the
width and clauses 9-12 TIE brow and chin to the nose. Those couplings are the whole design:
independent parameters allow impossible combinations, coupled ones do not. -/
def inRange (p : Params) : Prop :=
  450 ≤ p.skullW ∧ p.skullW ≤ 750 ∧
  300 ≤ p.noseProj ∧ p.noseProj ≤ 650 ∧
  -- EXACTLY the Python PARAM_RANGE, checked against it rather than approximated. A first
  -- version used 1.00-1.50 and 1.50-2.00 while Python allowed 0.95-1.55 and 1.45-2.15, so
  -- the proof did not cover every admissible parameter -- a silent gap between the theorem
  -- and the code it is about, which is the one failure a proof must not have.
  95 * p.skullW ≤ 100 * p.skullH ∧ 100 * p.skullH ≤ 155 * p.skullW ∧   -- 0.95x .. 1.55x
  145 * p.skullW ≤ 100 * p.faceH ∧ 100 * p.faceH ≤ 215 * p.skullW ∧    -- 1.45x .. 2.15x
  0 < p.browZ ∧ 100 * p.browZ ≤ 67 * p.noseProj ∧              -- brow stays behind the tip
  0 < p.chinZ ∧ 100 * p.chinZ ≤ 88 * p.noseProj                -- chin stays behind the tip

/-- Vertical positions AT 100x SCALE, so no division appears and omega decides directly.
The ratios match holographic_headspec exactly: mouth at 0.66 and nose at 0.36 of the face
height below the eye line, brow at 0.26 and crown at 1.00 of the skull height above it. -/
def chinY100  (p : Params) : Int := -100 * p.faceH
def mouthY100 (p : Params) : Int := -66 * p.faceH
def noseY100  (p : Params) : Int := -36 * p.faceH
def eyeY100   (_ : Params) : Int := 0
def browY100  (p : Params) : Int := 26 * p.skullH
def crownY100 (p : Params) : Int := 100 * p.skullH

theorem faceH_pos (p : Params) (h : inRange p) : 0 < p.faceH := by
  obtain ⟨h1, _, _, _, _, _, h7, _, _, _, _, _⟩ := h; omega

theorem skullH_pos (p : Params) (h : inRange p) : 0 < p.skullH := by
  obtain ⟨h1, _, _, _, h5, _, _, _, _, _, _, _⟩ := h; omega

/-- **VERTICAL ORDERING** -- crown > brow > eye > nose > mouth > chin, for EVERY admissible
parameter vector. This is the invariant a pancake violates and the one an unconstrained fit
kept violating. -/
theorem vertical_order (p : Params) (h : inRange p) :
    chinY100 p < mouthY100 p ∧ mouthY100 p < noseY100 p ∧ noseY100 p < eyeY100 p ∧
    eyeY100 p < browY100 p ∧ browY100 p < crownY100 p := by
  have hf := faceH_pos p h
  have hs := skullH_pos p h
  unfold chinY100 mouthY100 noseY100 eyeY100 browY100 crownY100
  refine ⟨by omega, by omega, by omega, by omega, by omega⟩

/-- **THE NOSE IS FRONTMOST.** Neither brow nor chin can pass the nose tip, because both are
bounded by 0.9x the nose projection. Before these were coupled, 292/400 random parameter
vectors put the chin or the brow IN FRONT of the nose -- an anatomically impossible head that
the parameterisation happily expressed. -/
theorem nose_frontmost (p : Params) (h : inRange p) :
    p.browZ < p.noseProj ∧ p.chinZ < p.noseProj := by
  obtain ⟨_, _, h3, _, _, _, _, _, h9, h10, h11, h12⟩ := h
  exact ⟨by omega, by omega⟩

/-- **ASPECT RATIO IS BOUNDED**, which is what makes a PANCAKE UNREACHABLE rather than merely
rejected: total head height lies strictly between 1x and 2.4x the FULL width, so no optimiser
can flatten the head however its objective is shaped. -/
theorem aspect_bounded (p : Params) (h : inRange p) :
    100 * (2 * p.skullW) < crownY100 p - chinY100 p ∧
    10 * (crownY100 p - chinY100 p) < 24 * (100 * (2 * p.skullW)) := by
  obtain ⟨h1, h2, _, _, h5, h6, h7, h8, _, _, _, _⟩ := h
  unfold crownY100 chinY100
  exact ⟨by omega, by omega⟩

/-- **NON-DEGENERACY.** Positive width and positive extents, so the head cannot collapse to a
plane or a point -- the other way a fit can "succeed" at nothing. -/
theorem non_degenerate (p : Params) (h : inRange p) :
    0 < p.skullW ∧ 0 < p.faceH ∧ 0 < p.skullH ∧ 0 < p.noseProj := by
  have hf := faceH_pos p h
  have hs := skullH_pos p h
  obtain ⟨h1, _, h3, _, _, _, _, _, _, _, _, _⟩ := h
  exact ⟨by omega, hf, hs, by omega⟩

/-- **THE HEAD IS A HEAD.** All four invariants at once -- the single statement that says the
parameterisation cannot express a non-head, which is the property that makes any future fit
safe regardless of its objective. -/
theorem is_a_head (p : Params) (h : inRange p) :
    (chinY100 p < mouthY100 p ∧ mouthY100 p < noseY100 p ∧ noseY100 p < eyeY100 p ∧
     eyeY100 p < browY100 p ∧ browY100 p < crownY100 p)
    ∧ (p.browZ < p.noseProj ∧ p.chinZ < p.noseProj)
    ∧ (100 * (2 * p.skullW) < crownY100 p - chinY100 p)
    ∧ (0 < p.skullW ∧ 0 < p.faceH ∧ 0 < p.skullH ∧ 0 < p.noseProj) :=
  ⟨vertical_order p h, nose_frontmost p h, (aspect_bounded p h).1, non_degenerate p h⟩

end LeCore.HeadSpec
