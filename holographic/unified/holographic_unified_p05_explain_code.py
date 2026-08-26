"""Part 05 of UnifiedMind's faculty surface -- 146 methods, explain_code .. mesh_split_edge.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, which grew to 17.4k lines
in one file and went past the 1 MB cap an agent can read in a single pass -- so the engine could no
longer read its own central nervous system. The class is assembled from these parts by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

Every method here is a real attribute of UnifiedMind at runtime (mixin, not delegation), so `mind.x()`,
`dir(mind)`, the doc generators and the service's tool introspection all behave exactly as before. The
bodies were moved by line range, not regenerated, so they are byte-identical to the originals.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access. The leading underscore on
the class name says so, and the reachability audit reads them as referenced-by-unified, not as
standalone capabilities.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind
from holographic.unified import check_part


class _UnifiedPart05:

    def explain_code(self, src, dialect="python"):
        """Explain source in plain English -- DETERMINISTICALLY (C1): same code, same sentences, every
        time. Four labeled layers per function, each derived from a distinct static analysis: signature,
        per-variable data flow (computed-from / read-count / feeds-return), a control-flow census, and an idiom
        layer that is the ONLY one allowed to speak of purpose -- and only on a shape-catalog match (names and
        constants blanked, so iq's rounded box with different half-extents is still recognized). An unmatched
        function gets 'not recognized', never a guess -- but a min()/max() of registered primitives IS recognized
        as a named union/intersection/subtraction (C6 composition layer). Returns {summary, functions, text}.
        `dialect` extends this beyond Python (C4): pass c_f64 | c_f32 | wgsl | js | zig_f64 | zig_f32 and the
        source is first parsed back to the shared IR (holographic_codeparse), then verbalized by the SAME
        verbalizer -- seven languages, one explainer, no per-language twin to drift.
        See holographic_codeverbal.verbalize."""
        from holographic.io_and_interop.holographic_codeverbal import verbalize
        if str(dialect) not in ("python", "c_f64", "c_f32", "wgsl", "js", "zig_f64", "zig_f32"):
            # a language we have no parser for -> honest triage (C5), never a guessed explanation
            from holographic.io_and_interop.holographic_codetriage import triage
            return triage(str(src))
        if str(dialect) != "python":
            from holographic.io_and_interop.holographic_codeparse import parse_kernel, ParseError
            try:
                return verbalize(parse_kernel(str(src), str(dialect)))
            except ParseError:
                # in-grammar dialect but the source is outside the kernel grammar -> triage rather than refuse
                from holographic.io_and_interop.holographic_codetriage import triage
                return triage(str(src))
        return verbalize(str(src))

    def register_code_idiom(self, name, example_src, purpose, citation=""):
        """Grow the idiom catalog (C6): register a purpose-recognizer by EXAMPLE -- the example function's
        blanked AST shape becomes the matcher. Returns the shape key so a test can assert recognition
        immediately. Additive; real published sources only in `citation` (panel discipline).
        See holographic_codeverbal.register_idiom."""
        from holographic.io_and_interop.holographic_codeverbal import register_idiom
        return register_idiom(str(name), str(example_src), str(purpose), str(citation))

    def register_composition_primitive(self, name, distance_expr_src):
        """Grow C6's COMPOSITION recognizer: register a primitive by its distance EXPRESSION (e.g. a sphere's
        'sqrt(px*px+py*py+pz*pz) - r') so that a min()/max() of such primitives is explained as a named union /
        intersection / subtraction rather than 'not recognized'. This is what closes the describe->emit->explain
        loop for COMPOSITE scenes. See holographic_codeverbal.register_composition_form."""
        from holographic.io_and_interop.holographic_codeverbal import register_composition_form
        return register_composition_form(str(name), str(distance_expr_src))

    def zig_dispatch_policy(self, n, calls_expected, min_calls_to_compile=3, min_n=4096):
        """Z5: which backend (numpy | zig) the native-kernel dispatcher would choose for arrays of length `n`
        called `calls_expected` times, WITH the reason -- the policy is data, sized to the measured 2-5x regime
        and the ~1-2 s first-call compile. Compose with mind.zig_batch_eval to act on the answer.
        See holographic_zigrun.dispatch_policy (AutoKernel enforces the same policy in-process, identity-gated:
        a native result that is not bit-identical to numpy in safe mode refuses the substitution permanently)."""
        from holographic.io_and_interop.holographic_zigrun import dispatch_policy
        return dispatch_policy(int(n), int(calls_expected), int(min_calls_to_compile), int(min_n))

    def zig_march_compare(self, kernel=None, width=96, height=72, max_steps=96, out_dir=None):
        """Z4's executed bar: sphere-trace the SAME rays through the engine's Python marcher and a natively
        compiled Zig loop (same scene SDF text, shared dialect table), shade both with the SAME code, and return
        {t_max_abs_diff, hit_flips, bit_identical, frames_byte_identical}. MEASURED verdict on the demo scene:
        f64 BIT-IDENTICAL, frames byte-identical, zig 3.8x on 110k rays x 96 steps -- and safe-vs-fast is a wash,
        so determinism costs nothing here. Pass out_dir to also write both PPM frames.
        See holographic_zigmarch.render_compare."""
        from holographic.io_and_interop.holographic_zigmarch import DEMO_SCENE, render_compare
        return render_compare(kernel if kernel is not None else DEMO_SCENE, width=int(width),
                              height=int(height), max_steps=int(max_steps), out_dir=out_dir)

    # -- C3: canonical element + delta chain (instancing, generalised) --------------------------------------
    def canonical_form(self, V, family="similarity"):
        """Split an element into (canonical, delta) with `V = canonical @ A.T + b`, EXACTLY (1e-12). `family` is
        rigid | similarity | affine, and choosing it is the whole decision: too small and nothing is recognised,
        too large and the delta costs more than the thing it describes.
        See holographic_canonmesh.canonical_form."""
        from holographic.mesh_and_geometry.holographic_canonmesh import canonical_form as _cf
        return _cf(np.asarray(V, float), family=str(family))

    def recognize_elements(self, elements, family="similarity", tol=6):
        """Instancing GENERALISED: two elements share a class when they are the same thing modulo the family's
        deltas, not when they happen to be the same array. Returns {canonicals, instances}.
        See holographic_canonmesh.recognize."""
        from holographic.mesh_and_geometry.holographic_canonmesh import recognize
        return recognize([np.asarray(V, float) for V in elements], family=str(family), tol=int(tol))

    def canon_storage_report(self, elements, family="similarity", tol=6):
        """{classes, raw_floats, total_floats, ratio, zlib_ratio, beats_zlib}. The honest baseline is zlib on the raw
        vertex buffer, which manages ~1.04x on float64 coordinates. KEPT NEGATIVE: a TRIANGLE cannot pay -- its hull
        is rank 2, so an affine delta is 9 floats for a 9-float triangle. The dividend scales with the ELEMENT
        against an O(1) delta: 0.75x at 3 vertices, 143x at 2000. Per-triangle canonicalisation is a RECOGNISER, not
        a compressor; its dividend is the C1 compute cache. See holographic_canonmesh.storage_report."""
        from holographic.mesh_and_geometry.holographic_canonmesh import storage_report
        return storage_report([np.asarray(V, float) for V in elements], family=str(family), tol=int(tol))

    # -- the tower's CEILING: projective is not "affine plus a bit" -----------------------------------------
    def is_affine_matrix(self, M, tol=1e-12):
        """Does the 4x4 `M` fix the plane at infinity -- is its bottom row [0,0,0,*]? A BOOLEAN, not a tolerance on
        a distance: a perspective that is *nearly* affine still divides, and the divide is the whole difference.
        See holographic_projectivetower.is_affine."""
        from holographic.mesh_and_geometry.holographic_projectivetower import is_affine
        return is_affine(np.asarray(M, float), tol=float(tol))

    def compose_word(self, chain):
        """Compose a chain of 4x4 transforms into ONE. **A group is CLOSED, so the "word" IS a "letter"** -- drawn
        from the same set as the generators it was built from. That is exactly why DL11's edit history collapses to
        a single group element rather than a sequence. See holographic_projectivetower.compose_word."""
        from holographic.mesh_and_geometry.holographic_projectivetower import compose_word as _cw
        return _cw([np.asarray(M, float) for M in chain])

    def affine_normality(self, t=(0.3, -0.7, 0.2), seed=0):
        """**Where the transform tower's mechanism stops.** {in_affine, in_projective, conjugate_is_affine}.
        Conjugating a translation by a ROTATION gives T(A t) to 1.1e-16 -- the ideal is normal in Aff. Conjugating
        it by a PERSPECTIVE gives a matrix that is not a translation and NOT EVEN AFFINE. So Aff is a subgroup of
        PGL and not a NORMAL one, and no delta can be pushed through a perspective the way it is pushed through a
        rotation. See holographic_projectivetower.affine_normality."""
        from holographic.mesh_and_geometry.holographic_projectivetower import affine_normality as _an
        return _an(t=tuple(t), seed=int(seed))

    def texture_projection_error(self, depths=(1.0, 4.0, 1.5), n=2000, seed=0):
        """The ceiling, in a renderer. Interpolating (u,v) linearly in SCREEN space assumes the triangle-to-texture
        map is affine; under perspective it is not. MEASURED at depths (1, 4, 1.5): affine max error **0.3310** --
        a third of the texture -- against **2.2e-16** for the homogeneous (u/w, v/w, 1/w) divide. The extra `q` is
        not another letter in the alphabet; it is an extra COORDINATE that enlarges the space the alphabet acts on,
        and by doing so breaks the affine group's normality. See holographic_projectivetower.texture_projection_error."""
        from holographic.mesh_and_geometry.holographic_projectivetower import texture_projection_error as _te
        return _te(depths=tuple(depths), n=int(n), seed=int(seed))

    # -- the transform TOWER: which layer of the affine group a thing lives in -----------------------------
    def classify_transform(self, fn, dim=3, n=64, tol=1e-9, seed=0):
        """**THE ONE ENTRY POINT: which floor of the transform tower is this on?** Give it any callable on points;
        it MEASURES and returns {layer, name, diagonalisable, bankable, delta_pushable, why}.

        1 = translation (the abelian ideal): one Fourier spectrum represents it, so it is a bind and it goes in a
        TransformBank. 2 = rotation / shear (the sl(n) peers): not a bind, but a delta still pushes through by
        conjugation, because the ideal is NORMAL. 3 = scale (the centre of GL): commutes with the whole linear part
        and not with translation. 4 = beyond the affine ceiling (projective, or not a group action): NO delta pushes
        through, because Aff is not normal in PGL.

        `delta_pushable` is the question the tower exists to answer -- it is `shade_adjoint`'s licence, DL11's
        closure, and the equivariance table's shape, in one boolean. See holographic_grouptower.classify_transform."""
        from holographic.mesh_and_geometry.holographic_grouptower import classify_transform as _ct
        return _ct(fn, dim=int(dim), n=int(n), tol=float(tol), seed=int(seed))

    def hypervector_layer(self):
        """**A hypervector used as an operator is ALWAYS the abelian ideal, and the algebra forbids anything else.**
        `bind` is a circular convolution, hence commutative; a convolution algebra can only represent an abelian
        group. `Hypervector.transform_layer()` says the same thing on the object.
        See holographic_grouptower.hypervector_layer."""
        from holographic.mesh_and_geometry.holographic_grouptower import hypervector_layer as _hl
        return _hl()

    def commutator_table(self):
        """Every claim of the transform tower, as a number. `Aff(n) = GL(n) |x| R^n`: translations are the abelian
        IDEAL ([T,T'] = 0), scale is CENTRAL in the linear part ([S,R] = [S,Sh] = 0) and NOT in the affine group
        ([S,T] = 0.49, because it acts on the ideal), and rotation-vs-shear are non-commuting PEERS ([R,Sh] = 0.23).
        In 2-D the rotations commute with each other -- SO(2) is abelian -- so the peers only become
        rotation-vs-rotation in 3-D ([Rx,Ry] = 0.50). See holographic_grouptower.commutator_table."""
        from holographic.mesh_and_geometry.holographic_grouptower import commutator_table
        return commutator_table()

    def semidirect_law(self, linear, t):
        """`max |A T(t) A^-1 - T(A t)|` -- zero, because the ideal is NORMAL. **One line, three costumes**: it is
        `shade_adjoint`'s 'push the delta onto the other operand'; it is why DL11's affine chain collapses to one
        group element; and it is why the equivariance table has the shape it has.
        See holographic_grouptower.semidirect_law."""
        from holographic.mesh_and_geometry.holographic_grouptower import semidirect_law as _sl
        return float(_sl(np.asarray(linear, float), np.asarray(t, float)))

    def is_diagonalisable(self, transform, seed=0):
        """Can ONE Fourier spectrum represent `transform`? Fit it on an encoded point, apply to another, return the
        relative error. Near zero -> it is a BIND and belongs in a TransformBank. **Only the abelian IDEAL is
        diagonalisable**, because a convolution algebra is commutative: translation 3.8e-16, rotation 5.4e-01,
        scale 1.3e-01. See holographic_grouptower.diagonalisable."""
        from holographic.mesh_and_geometry.holographic_grouptower import diagonalisable
        return float(diagonalisable(transform, seed=int(seed)))

    def mellin_promotes_scale(self, n=1024, s=1.4, seed=0):
        """**A layer you cannot diagonalise, you relocate.** On a LOG axis a dilation becomes a translation, joins
        the abelian ideal, and becomes a bind. Returns {linear_axis, log_axis} relative errors: ~2.8 against 1e-15.
        That is Reddy-Chatterji, and `mellin_scale` is the engine's use of it.
        See holographic_grouptower.mellin_promotes_scale."""
        from holographic.mesh_and_geometry.holographic_grouptower import mellin_promotes_scale as _mp
        return _mp(n=int(n), s=float(s), seed=int(seed))

    # -- a prebuilt map of hypervector transforms: a group representation, not a lookup table ----------------
    def transform_bank(self, dim=None, seed=0):
        """A prebuilt map of named hypervector transforms, held as their Fourier spectra. `add_random_unitary`,
        `add_rotation(k)`, `apply`, `apply_batch`, `apply_chain`, `power`, `inverse_spectrum`, `stats`.

        CACHING A SPECTRUM SAVES 28% (1.42x). **COMPOSITION IS THE PAYOFF**: circular convolution is diagonal in the
        Fourier basis, so a CHAIN of k transforms is the PRODUCT of their spectra and collapses into ONE bind --
        13.5x on a chain of 8, exact to 5.7e-17. SCALE IS NOT IN THE BANK: a dilation is not shift-invariant, so no
        spectrum represents it (fit one on a vector, apply it to another: relative error 1.579). That is DL11's
        finding; the Mellin lift makes scale a SHIFT on a log axis, which is a different bank over a different axis.
        See holographic_transformbank.TransformBank."""
        from holographic.caching_and_storage.holographic_transformbank import TransformBank
        return TransformBank(int(dim if dim is not None else self.dim), seed=int(seed))

    def scale_is_not_a_bind(self, dim=256, s=1.5, seed=0):
        """MEASURE, do not assume, that a dilation is not diagonal in the Fourier basis: fit its 'spectrum' on one
        vector, apply it to another, return the relative error. 1.579 -- the wrong object, not a lossy fit.
        See holographic_transformbank.scale_is_not_a_bind."""
        from holographic.caching_and_storage.holographic_transformbank import scale_is_not_a_bind as _s
        return float(_s(dim=int(dim), s=float(s), seed=int(seed)))

    # -- C1/C4: the dependency-key layer -- key on what the operator READS ----------------------------------
    def delta_cache(self, op, canonical, policy="read_set", op_name=None, transform_family=None):
        """A cache whose key is (operator, canonical class, the delta-IDs the operator READS). A material delta
        cannot change a geometric quantity, so it never enters the key. `policy="equivariant"` additionally drops the
        shape delta when the operator is MEASURED invariant under that delta's family (see mind.equivariance_table).
        REFUSES a non-deterministic evaluator -- a cache serves its first draw forever while the uncached path keeps
        drawing (backlog D1/C4). See holographic_deltacache.DeltaCache."""
        from holographic.caching_and_storage.holographic_deltacache import DeltaCache
        return DeltaCache(op, np.asarray(canonical, float), policy=str(policy),
                          op_name=op_name, transform_family=transform_family)

    def delta_cache_report(self, op, canonical, scene, deltas, op_name=None, transform_family=None):
        """Every keying policy against the brute baseline: {policy: {computes, speedup, max_abs_diff,
        bit_identical}}. MEASURED on 400 triangles (rotations x materials): brute 400 computes, read_set 50 (8.0x,
        BIT-IDENTICAL), equivariant 1 (400x). KEPT NEGATIVE: the equivariant path is NOT bit-identical (8.3e-17) --
        rotating a triangle and re-integrating accumulates round-off the canonical evaluation never incurs, so the
        CACHE is the one that is right, and `max_abs_diff` is reported rather than a boolean.
        See holographic_deltacache.cache_report."""
        from holographic.caching_and_storage.holographic_deltacache import cache_report
        return cache_report(self._as_operator(op), np.asarray(canonical, float), list(scene), list(deltas),
                            op_name=op_name, transform_family=transform_family)

    def evaluate_elements(self, elements, op, op_name, family="similarity", tol=6):
        """**Part C, end to end.** Hand it RAW point sets with no shape ids: `canonmesh.recognize` derives the
        canonical classes (C3), `equivariance_table` says what the operator reads (C2), and the cache keys on that
        (C1). Returns (values, {classes, computes, verdict, family}). MEASURED on 200 raw triangles from 5 base
        shapes: `area` under `similarity` is 5 classes, 5 computes, exact to 1.4e-14 -- 40x. A COMPOSITE FAMILY'S
        VERDICT IS THE WEAKEST OF ITS PARTS (`area` is invariant under `rigid`, equivariant under `similarity`,
        because a uniform scale moves it), and RECOGNITION ALONE IS NOT ENOUGH: reusing the canonical's area
        directly under `similarity` is wrong by 8.54. See holographic_deltacache.evaluate_elements."""
        from holographic.caching_and_storage.holographic_deltacache import evaluate_elements as _ee
        return _ee([np.asarray(e, float) for e in elements], self._as_operator(op), str(op_name),
                   family=str(family), tol=int(tol))

    def family_verdict(self, op_name, family):
        """The verdict for a composite canonicalisation family (`rigid` | `similarity` | `affine`): the WEAKEST of
        its elementary parts, each measured by `classify_equivariance`. Taking the strongest instead would produce a
        cache that reuses a value the delta actually changed. See holographic_deltacache.family_verdict."""
        from holographic.caching_and_storage.holographic_deltacache import family_verdict as _fv
        return _fv(str(op_name), str(family))

    def is_deterministic(self, fn, sample, trials=3):
        """Does `fn` return a bit-identical result for a repeated input? The gate a cache needs, and D1's
        coordinate-keyed-sampling rule one level up. See holographic_deltacache.is_deterministic."""
        from holographic.caching_and_storage.holographic_deltacache import is_deterministic as _id
        return bool(_id(fn, sample, trials=int(trials)))

    # -- DL11: canonical affine recovery (Fourier-Mellin coarse + continuous refine) ------------------------
    def recover_affine(self, before, after, refine=True):
        """Recover the canonical (S, T) of an arbitrary translate/scale edit history: `after(x) = before((x-T)/S)`.
        A chain of such edits does not commute, but it CLOSES -- every order collapses to some single group element,
        which is the recoverable object. Coarse Fourier-Mellin (|FFT| kills the translation; a log-frequency
        resample turns the dilation into a SHIFT, and the estimator is `est_dx` again) then a shrinking-grid refine
        on the (s, t) manifold. Returns {scale, shift, alignment, coarse_scale}. MEASURED: 3.7e-04 scale error on a
        4-edit chain at alignment 0.9995. See holographic_registration.recover_affine."""
        from holographic.sampling_and_signal.holographic_registration import recover_affine as _ra
        return _ra(np.asarray(before, float), np.asarray(after, float), refine=bool(refine))

    def affine_compose(self, chain):
        """Collapse a chain of (s, t) edits into ONE canonical (S, T) by the affine group law -- exact, not an
        approximation. `x -> s2(s1 x + t1) + t2 = (s2 s1) x + (s2 t1 + t2)`. Order changes WHICH (S, T) you get;
        every order gives some single (S, T). See holographic_registration.affine_compose."""
        from holographic.sampling_and_signal.holographic_registration import affine_compose as _ac
        return _ac(list(chain))

    def mellin_scale(self, before, after, use_log=True):
        """Recover the DILATION alone, via the Fourier-Mellin lift. KEPT NEGATIVE: the SUPPORT BAND is the gate,
        not log-vs-plain. A constant amplitude factor cannot move a correlation's argmax (verified), so the
        backlog's stated reason for log magnitudes is not the operative one -- but on a narrowband spectrum `log`
        amplifies the noise floor and the peak pins at zero shift for every true scale.
        See holographic_registration.mellin_scale."""
        from holographic.sampling_and_signal.holographic_registration import mellin_scale as _ms
        return float(_ms(np.asarray(before, float), np.asarray(after, float), use_log=bool(use_log)))

    # -- F1: LSCM -- the angle-preserving unwrap, and the metric that sees folds -----------------------------
    def mesh_lscm(self, mesh, pins=None):
        """LEAST-SQUARES CONFORMAL MAP (Levy et al., SIGGRAPH 2002): the angle-preserving unwrap, as ONE linear
        least-squares solve on the mesh -- no iteration, no autodiff. Exact on a developable surface (quasi-conformal
        ratio 1.000000) and best on angle everywhere measured. KEPT NEGATIVE: it buys angles with AREA (0.4420 spread
        on a hemisphere cap against isomap's 0.2957) and it does not guarantee a FOLD-FREE map. See
        holographic_meshuv.lscm."""
        from holographic.mesh_and_geometry.holographic_meshuv import lscm
        return lscm(self._as_mesh(mesh), pins=pins)

    def mesh_uv_angle_distortion(self, mesh, uv):
        """The metric LSCM optimises: per-face quasi-conformal ratio sigma1/sigma2. Returns {mean, median, max,
        flipped, n_faces}; 1.0 is conformal. **Report the MEDIAN** -- the mean is unbounded (one near-degenerate
        face sends it to infinity: 398.0 mean against a 4.8 median on a stretched cap). **And report `flipped`** --
        neither the stretch metric nor the mean ratio can see a FOLD, and a fold is a MINORITY orientation, not a
        negative determinant (a globally mirrored chart has every det < 0 and no folds at all).
        See holographic_meshuv.uv_angle_distortion."""
        from holographic.mesh_and_geometry.holographic_meshuv import uv_angle_distortion
        return uv_angle_distortion(self._as_mesh(mesh), np.asarray(uv, float))

    def mesh_uv_area_distortion(self, mesh, uv):
        """The log-spread of (UV face area / 3-D face area); 0 = area-preserving. The price a conformal map pays.
        See holographic_meshuv.uv_area_distortion."""
        from holographic.mesh_and_geometry.holographic_meshuv import uv_area_distortion
        return uv_area_distortion(self._as_mesh(mesh), np.asarray(uv, float))

    def mesh_uv_report(self, mesh, methods=("lscm", "isomap", "planar")):
        """Every chart on every metric -- {method: {angle, area, stretch}} -- so nobody compares two charts on a
        functional only one of them optimises. See holographic_meshuv.uv_report."""
        from holographic.mesh_and_geometry.holographic_meshuv import uv_report
        return uv_report(self._as_mesh(mesh), methods=tuple(methods))

    # -- F8: the brain/muscle format contract -- rank-ordered TT cores as a progressive LOD stream ----------
    def stream_encode(self, X, tol=1e-10, max_rank=None):
        """Encode a field as a PROGRESSIVE payload: {descriptor, levels}. Every byte prefix is itself a valid,
        coarser field. The descriptor is plain data a front end reads BEFORE fetching: shape, dtype, full_ranks,
        per-level `bytes`, `rel_rms`, `rel_max`, and `monotone_in`. That is the whole contract -- the brain
        publishes what each prefix costs and what it is worth; the muscle chooses.
        See holographic_stream.stream_encode."""
        from holographic.io_and_interop.holographic_stream import stream_encode as _se
        return _se(np.asarray(X, float), tol=float(tol), max_rank=max_rank)

    def stream_decode(self, payload, level=-1):
        """Reconstruct the field from one level. Every level decodes to the FULL shape -- a coarse level is a
        SMOOTHED field, not a small one. Rank is not resolution. See holographic_stream.stream_decode."""
        from holographic.io_and_interop.holographic_stream import stream_decode as _sd
        return _sd(payload, level=int(level))

    def stream_prefix(self, payload, max_bytes):
        """The richest level that fits in `max_bytes`, or None. Decided from the descriptor alone.
        See holographic_stream.stream_prefix."""
        from holographic.io_and_interop.holographic_stream import stream_prefix as _sp
        return _sp(payload, int(max_bytes))

    def stream_report(self, X, payload):
        """The ladder, with both norms: {levels, monotone_rms, monotone_max, dense_bytes, best_ratio_at_10pct}.
        THE GUARANTEE IS IN RMS, not max-abs: TT truncation is Frobenius-optimal, so adding a rank always lowers
        the L2 error and can still make one voxel worse -- measured, the max-abs error rises at 4 of 15 levels on
        white noise while the RMS falls at every one. A progressive format must publish which norm its monotonicity
        is in. See holographic_stream.stream_report."""
        from holographic.io_and_interop.holographic_stream import stream_report as _sr
        return _sr(np.asarray(X, float), payload)

    # -- C2: the equivariance table -- the whole cache policy, measured not declared -------------------------
    def equivariance_table(self, trials=12, tol=1e-9, seed=0):
        """MEASURE which of {invariant, equivariant, recompute} holds for each (operator, affine family) pair.
        Returns {operator: {transform: verdict}}. Regenerate it on your box; a table that cannot re-measure itself
        is a rumour. See holographic_equivariance.equivariance_table."""
        from holographic.mesh_and_geometry.holographic_equivariance import equivariance_table as _et
        return _et(trials=int(trials), tol=float(tol), seed=int(seed))

    def cache_policy(self, op_name, transform, trials=12, tol=1e-9, seed=0):
        """The cache-key consequence of the verdict: {verdict, key_includes_delta, read_set, note}. `invariant`
        means the delta leaves the key entirely; `equivariant` means one compute per canonical class, but the law
        needs the operator's READ-SET cached alongside it -- and every non-rigid law here reads the NORMAL.
        See holographic_equivariance.cache_policy."""
        from holographic.mesh_and_geometry.holographic_equivariance import cache_policy as _cp
        return _cp(str(op_name), str(transform), trials=int(trials), tol=float(tol), seed=int(seed))

    def classify_equivariance(self, op_name, transform, trials=12, tol=1e-9, seed=0):
        """The strongest verdict that holds, measured. A `recompute` result means the REGISTERED law failed -- which
        may mean no law exists, or that the law is wrong. Two cells of this table read `recompute` until the laws
        were derived properly: when you see recompute, derive before you believe.
        See holographic_equivariance.classify."""
        from holographic.mesh_and_geometry.holographic_equivariance import classify as _c
        return _c(str(op_name), str(transform), trials=int(trials), tol=float(tol), seed=int(seed))

    def shade_adjoint(self, tri, light, A):
        """The ADJOINT move done correctly for ANY affine: shade(A x, L) == max(0, n . (A^-1 L) / ||A^-T n||).
        Part C states it as 'unrotate the light', shade(A x, L) == shade(x, A^-1 L) -- exact for a ROTATION
        (3.9e-16) and wrong for everything else, including a plain uniform scale, by 0.38. The factor above is what
        the naive statement drops. See holographic_equivariance.shade_adjoint."""
        from holographic.mesh_and_geometry.holographic_equivariance import shade_adjoint as _sa
        return _sa(np.asarray(tri, float), np.asarray(light, float), np.asarray(A, float))

    # -- F4: the cloud stack -- the closed form pays on the SHADOW ray ---------------------------------------
    def cloud_transmittance(self, volume, O, D, L, sigma_t=1.0):
        """Beer-Lambert transmittance exp(-sigma_t * tau) with `tau` from ONE closed-form integral -- O(1) in march
        steps, because there are none. `L` is scalar or PER-RAY (R,), and it matters: a shadow ray's length to the
        medium boundary varies per sample, and passing a median instead costs three orders of magnitude of accuracy.
        See holographic_cloud.transmittance."""
        from holographic.rendering.holographic_cloud import transmittance as _t
        return _t(volume, np.asarray(O, float), np.asarray(D, float), L, sigma_t=float(sigma_t))

    def cloud_single_scatter(self, volume, O, D, L, sun_dir, ceiling, view_steps=32, shadow_steps=0,
                             sigma_t=1.0, sigma_s=0.9, g=0.4, integrate="rect"):
        """Single-scattered radiance, returning (radiance, density_evals). The VIEW ray must march -- its integrand
        contains the transmittance being accumulated -- but every SHADOW ray is one closed-form integral.
        MEASURED (64 rays, 32 view steps): 32 density evaluations against a 64-step marched shadow's 2,080 -- 65x
        fewer, 52x faster, and 13x MORE accurate (3.03e-07 vs 3.94e-06), because the closed form is the exact
        integral and the march is the one carrying error.
        `integrate="analytic"` (opt-in; default "rect" is bit-identical to before) integrates each segment against
        its OWN extinction instead of assuming constant transmittance across the step -- Hillaire's Frostbite
        SIGGRAPH 2015 form, Sint = (S - S exp(-sigma_e dt))/sigma_e. MEASURED: 8-step error 6.29e-03 -> 5.38e-05,
        which beats even the 64-step rectangle rule (7.22e-04) -- an ~8x cut in density evals at better accuracy.
        The advantage shrinks as extinction rises (a dense medium self-terminates inside one step).
        See holographic_cloud.single_scatter."""
        from holographic.rendering.holographic_cloud import single_scatter as _ss
        return _ss(volume, np.asarray(O, float), np.asarray(D, float), float(L), sun_dir, float(ceiling),
                   view_steps=int(view_steps), shadow_steps=int(shadow_steps),
                   sigma_t=float(sigma_t), sigma_s=float(sigma_s), g=float(g), integrate=integrate)

    def cloud_report(self, volume, O, D, L, sun_dir, ceiling, view_steps=32, reference_shadow_steps=64):
        """The bar, carried with the capability: {evals_closed, evals_reference, eval_ratio, max_error, mean_error}.
        The reference is a HEAVILY marched shadow -- a coarse one would flatter the closed form by being wrong in the
        same direction. See holographic_cloud.cloud_report."""
        from holographic.rendering.holographic_cloud import cloud_report as _cr
        return _cr(volume, np.asarray(O, float), np.asarray(D, float), float(L), sun_dir, float(ceiling),
                   view_steps=int(view_steps), reference_shadow_steps=int(reference_shadow_steps))

    # -- F3: the points -> SDF -> mesh path (the inverse of sdf_surface_points) -----------------------------
    def sdf_from_points(self, points, normals, lo, hi, res, chunk=4096):
        """A signed distance grid from an ORIENTED point cloud: distance to the nearest sample, signed by that
        sample's normal. ACCURACY IS SET BY THE CLOUD, NOT THE GRID -- the near-surface error tracks the sample
        spacing at ~1.3-1.7x, so refining the grid under a sparse cloud buys nothing. `chunk` bounds the distance
        matrix; without it a 24^3 grid against 9,600 points allocates 133M floats and the process is killed.
        See holographic_isosurface.sdf_from_points."""
        from holographic.mesh_and_geometry.holographic_isosurface import sdf_from_points as _sp
        return _sp(np.asarray(points, float), np.asarray(normals, float), lo, hi, int(res), chunk=int(chunk))

    def surface_nets(self, field, grids, iso=0.0):
        """DUAL isosurface extraction: (vertices, quads) from a scalar field. One vertex per sign-changing cell at
        the mean of its edge crossings; one quad per sign-changing grid edge. A closed surface comes out WATERTIGHT.
        HONEST SCOPE: naive surface nets, not Dual Contouring -- averaging the crossings rounds off SHARP features,
        which DC's QEF solve recovers. Smooth surfaces only. See holographic_isosurface.surface_nets."""
        from holographic.mesh_and_geometry.holographic_isosurface import surface_nets as _sn
        return _sn(np.asarray(field, float), list(grids), iso=float(iso))

    def points_to_mesh(self, points, normals, lo, hi, res, chunk=4096):
        """The whole path: oriented points -> SDF grid -> watertight quad mesh. Returns (verts, quads, field, grids)
        so the intermediate field can be inspected rather than trusted. MEASURED on a unit sphere (600 samples,
        32^3 grid, cell 0.1032): 1,804 verts, 1,802 quads, watertight, max vertex error 0.0454 = 0.44 cells -- and
        the MESH is 4.7x more accurate than the FIELD it came from, because averaging the twelve edge crossings
        cancels per-sample noise. See holographic_isosurface.points_to_mesh."""
        from holographic.mesh_and_geometry.holographic_isosurface import points_to_mesh as _pm
        return _pm(np.asarray(points, float), np.asarray(normals, float), lo, hi, int(res), chunk=int(chunk))

    def mesh_is_oriented(self, quads):
        """Is every DIRECTED edge traversed exactly once? The property a half-edge structure, a normal, and any
        field solver actually need -- and the one `watertight` cannot see. A watertight-but-unoriented mesh has each
        undirected edge in two faces that wind the SAME way, so half its normals point inward.
        See holographic_isosurface.is_oriented."""
        from holographic.mesh_and_geometry.holographic_isosurface import is_oriented
        return bool(is_oriented(np.asarray(quads, int)))

    def mesh_report(self, mesh_or_vertices, quads=None, sdf=None):
        """MESH REPORT, dispatching on the input (this method was DEFINED TWICE after a branch merge -- the
        isosurface form was silently shadowed by the later meshtools form, caught by the isosurface wiring
        test; merged here so BOTH capabilities stay callable, neither reimplemented):
        - mesh_report(mesh): the general topology + shape scoreboard -- verts, faces, quad/tri/ngon fraction,
          boundary/nonmanifold edges, manifold/closed flags, euler, valence histogram, bbox, centroid.
          See holographic_meshtools.mesh_report.
        - mesh_report(vertices, quads, sdf=...): EXTRACTION VALIDATION for an isosurfacer's raw output --
          {n_vertices, n_quads, watertight, euler, surface_error}; the honest surface_error baseline is the
          CELL SIZE (a dual extractor cannot place a vertex better than the cell it lives in).
          See holographic_isosurface.mesh_report."""
        if quads is not None:
            from holographic.mesh_and_geometry.holographic_isosurface import mesh_report as _mr
            return _mr(np.asarray(mesh_or_vertices, float), np.asarray(quads, int), sdf=sdf)
        from holographic.mesh_and_geometry.holographic_meshtools import mesh_report as _mr
        return _mr(self._as_mesh(mesh_or_vertices))

    def turnaround(self, mesh, ref_mesh=None, views=("top", "front", "side", "3q"), width=360, height=360):
        """TURNAROUND: render a mesh from the standard modelling views (top/front/side/3q) in one call and, if a
        ref_mesh is given, score how well the silhouettes MATCH per view (silhouette IoU in [0,1], 1.0 = identical
        outline). Returns {sheet, views, iou, mean_iou}. The critic loop that caught the mantis slurped legs, now a
        NUMBER an agent can optimise (raise mean_iou by fixing the lowest view). Pair with mesh_report for topology.
        See holographic_render.turnaround."""
        from holographic.rendering.holographic_render import turnaround as _ta
        return _ta(self._as_mesh(mesh), ref_mesh=(self._as_mesh(ref_mesh) if ref_mesh is not None else None),
                   views=views, width=width, height=height)


    # -- B1: fill the gaps in a field (harmonic / majority, dispatched on type) -----------------------------
    def inpaint(self, field, known, kind="auto", periodic=False):
        """Fill the unknown cells of a field. `known` is a boolean mask, True where the value is trusted.
        `kind="auto"` sends an integer array to a majority vote and a float array to a harmonic (Laplace) solve --
        a discrete field has no mean, so averaging it is a category error.

        A CONTINUOUS field may be (H,W) or (H,W,C): an RGB image fills in ONE call (per-channel, shared mask), so a
        caller stops writing its own 3x loop. HONEST: multi-channel is API ergonomics, NOT a speedup -- the solver
        is iterative (no factorisation to amortise), so the per-channel loop is measured-fastest and is what runs.

        MEASURED (48x48, 59% erased, 8 seeds): harmonic MAE 0.0015 mean; majority accuracy 0.9653 mean, and 0.9990
        in region interiors -- nearly all the error is boundary error. THE BOUNDARY CONDITION IS THE GATE:
        `periodic=False` (edge-clamped) is the default, because wrapping a non-periodic field with np.roll solves a
        different problem and costs 5.4x. See holographic_inpaint.inpaint."""
        from holographic.sampling_and_signal.holographic_inpaint import inpaint as _ip
        return _ip(np.asarray(field), np.asarray(known, bool), kind=kind, periodic=periodic)

    def harmonic_fill(self, field, known, periodic=False, iters=2000, tol=1e-9):
        """Fill a CONTINUOUS field's holes by a Laplace solve: each hole relaxes to the mean of its four neighbours,
        known cells pinned bit-for-bit. The minimal-energy interpolant. Accepts (H,W) or (H,W,C) -- an RGB field
        fills per-channel with the shared mask in one call (a convenience, not a speedup: the solver is iterative).
        See holographic_inpaint.harmonic_fill."""
        from holographic.sampling_and_signal.holographic_inpaint import harmonic_fill as _hf
        return _hf(np.asarray(field, float), np.asarray(known, bool), periodic=periodic, iters=iters, tol=tol)

    def majority_fill(self, labels, known, periodic=False, iters=1000):
        """Fill a CATEGORICAL field's holes by neighbour vote, one ring per sweep. Ties break toward the lowest
        label index -- stated, not accidental. See holographic_inpaint.majority_fill."""
        from holographic.sampling_and_signal.holographic_inpaint import majority_fill as _mf
        return _mf(np.asarray(labels), np.asarray(known, bool), periodic=periodic, iters=iters)

    def fill_report(self, truth, filled, known):
        """Score a fill ON THE HOLES ONLY: {n_holes, hole_fraction, mae, accuracy}. Scoring the known cells would
        flatter every method equally. See holographic_inpaint.fill_report."""
        from holographic.sampling_and_signal.holographic_inpaint import fill_report as _fr
        return _fr(np.asarray(truth), np.asarray(filled), np.asarray(known, bool))

    # -- F7: frame-to-frame motion by ONE UNBIND (TAA's reprojection velocity) ------------------------------
    def est_dx(self, a, b, normalize=False, subpixel=True):
        """The (dy, dx) translation of frame `b` relative to `a`, recovered by ONE unbind: cross-correlation in the
        Fourier domain is `conj(F(a)) * F(b)`, and its peak is the shift. Sub-pixel by parabolic interpolation.
        MEASURED on a real rendered frame warped by a known amount: 0.0705 px mean, 0.1087 px worst.
        KEPT NEGATIVE: `normalize=True` (textbook PHASE correlation) is 2.3x WORSE at sub-pixel -- it sharpens the
        peak toward a delta, which a parabola cannot fit. See holographic_reproject.est_dx."""
        from holographic.rendering.holographic_reproject import est_dx as _e
        return _e(np.asarray(a, float), np.asarray(b, float), normalize=normalize, subpixel=subpixel)

    def reproject(self, a, b, tile=None):
        """Predict frame `b` from frame `a` by warping it with the recovered motion. `tile=None` uses one global
        shift; an int uses a per-tile field. MEASURED on real 192x192 frames: a lateral pan gives 40.46 dB global
        against 36.67-40.67 dB tiled (tiling LOSES), while a dolly gives 34.82 dB global against 37.43 dB at tile 48
        (tiling WINS). Tiling pays only on a non-uniform flow field. See holographic_reproject.reproject."""
        from holographic.rendering.holographic_reproject import reproject as _r
        return _r(np.asarray(a, float), np.asarray(b, float), tile=tile)

    def reproject_report(self, a, b, tiles=(32, 48, 64)):
        """The comparison carried WITH the capability: {no_warp, global, tiled, uniformity, best}. `no_warp` is the
        honest baseline. `best` needs the TRUE next frame, so this is an OFFLINE mode-chooser, not a runtime gate --
        at runtime the camera knows how it moved. KEPT NEGATIVE: the per-tile shift SPREAD does not separate the
        regimes (a pure translation has the largest spread and global still wins by 22 dB), so F7's 'one unbind per
        tile INSTEAD of motion vectors from geometry' does not hold. See holographic_reproject.reproject_report."""
        from holographic.rendering.holographic_reproject import reproject_report as _rr
        return _rr(np.asarray(a, float), np.asarray(b, float), tiles=tuple(tiles))

    # -- K1/K2: code as canonical shape + name delta (exact decomposition, not a compressor) ----------------
    def code_decompose(self, node_or_src):
        """Split a statement into (shape_template, delta): the AST with every identity slot blanked, and the ordered
        names/attributes/constants that were in them. Part C's triangle, applied to code. MEASURED: 63,121 of 63,121
        statement subtrees across 421 modules reconstruct bit-exactly. See holographic_codestructure.decompose."""
        import ast as _ast
        from holographic.io_and_interop.holographic_codestructure import decompose
        node = _ast.parse(node_or_src).body[0] if isinstance(node_or_src, str) else node_or_src
        return decompose(node)

    def code_recompose(self, template, delta):
        """The exact inverse of code_decompose. Raises on a delta of the wrong length -- a silent short-read would
        produce plausible, wrong code. See holographic_codestructure.recompose."""
        from holographic.io_and_interop.holographic_codestructure import recompose
        return recompose(template, list(delta))

    def code_structure(self, src):
        """A module's top-level statements as (codebook, stream): shape_key -> template, and [(key, delta), ...].
        `mind.code_rebuild` inverts it -- MEASURED, 421/421 modules rebuild to a byte-identical normalized source.
        See holographic_codestructure.module_structure."""
        from holographic.io_and_interop.holographic_codestructure import module_structure
        return module_structure(str(src))

    def code_rebuild(self, codebook, stream):
        """Rebuild normalized source from (codebook, stream). "Normalized" is precise, not a hedge: `ast.unparse` is
        a FIXED POINT on all 421 modules here and the reparsed AST is identical, so only formatting and comments --
        which the AST never carried -- are discarded. See holographic_codestructure.rebuild_source."""
        from holographic.io_and_interop.holographic_codestructure import rebuild_source
        return rebuild_source(codebook, list(stream))

    def code_shape_census(self, src):
        """{statements, distinct_kept, distinct_erased, reuse_kept, reuse_erased, singleton_fraction} over a source's
        STATEMENT subtrees -- the unit the backlog's 2.36x is measured on (this tree: 2.34x erased vs 1.19x kept,
        83.2% singletons). At FUNCTION granularity the same census reads 1.13x. State the unit with the number.
        See holographic_codestructure.shape_census."""
        from holographic.io_and_interop.holographic_codestructure import shape_census, statements
        return shape_census(statements(str(src)))

    def code_byte_report(self, src):
        """The codec comparison, carried WITH the capability: {raw, zlib_raw, codebook_bytes, delta_bytes,
        structure_bytes, ratio_vs_zlib, beats_zlib}. On the whole tree the structure is 1.12x LARGER than zlib --
        an exact decomposition is not a compressor, and 83.2% of shapes occur exactly once.
        See holographic_codestructure.byte_report."""
        from holographic.io_and_interop.holographic_codestructure import byte_report
        return byte_report(str(src))

    # -- K3: content-keyed memoization of PURE functions (the purity gate is the point) ---------------------
    def memoize_pure(self, fn, cache=None, maxsize=128):
        """Memoize `fn` on (its exact canonical source, its arguments) -- and REFUSE if `fn` is not pure.
        `is_pure` rejects the clock, RNG, IO, global writes, and transitive impurity, so the cache cannot return a
        stale answer. Measured 36x on a repeated 256x256 SVD, bit-identical.

        KEYED ON THE EXACT SOURCE, NOT ON A CANONICAL SHAPE. The backlog calls this "shape-keyed memoization"; a
        shape erases constants, so `x + 1` and `x + 2` share one and would share a cache entry. KEPT NEGATIVE: the
        key costs O(input bytes) -- fingerprinting a 512x512 array costs 1.747 ms while `A.sum()` costs 0.084 ms,
        so a cheap function of a large array LOSES 21x. See holographic_pycontext.memoize_pure."""
        from holographic.io_and_interop.holographic_pycontext import memoize_pure as _mp
        return _mp(fn, cache=cache, maxsize=int(maxsize))

    def canonical_shape(self, fn_or_src, name=None):
        """The structural fingerprint of a function -- node types and nesting depth, identifiers and constants
        ERASED. A COMPRESSION primitive, never a cache key: `x + 1` and `x + 2` share a shape. Measured reuse: 1.13x
        at FUNCTION granularity and 2.34x at STATEMENT granularity -- the backlog's 2.36x is the statement number and
        it reproduces. State the unit with the number. See holographic_pycontext.canonical_shape and
        holographic_codestructure.shape_census."""
        from holographic.io_and_interop.holographic_pycontext import canonical_shape as _cs
        return _cs(fn_or_src, name=name)

    # -- G2: the GENERIC scatter -- any rank, any kernel, exact on demand ------------------------------------
    def scatter(self, points, values, shape, kernel="bilinear", periodic=False):
        """SCATTER = the BUNDLE: deposit each point's value onto a grid of any RANK through a kernel. `points`
        (N, D) in grid-cell units, `values` (N,) or (N, C). Rank-agnostic (verified 1-D through 4-D), mass
        preserving (a partition-of-unity kernel's weights sum to 1), `periodic=False` clamps at the edges.

        `kernel="nearest"` is the GPU's scatter -- an atomic add at an index, ties rounding up -- and scattering
        ones at integer coordinates IS `np.bincount`. `scatter_to_field` / `scatter_to_field_3d` are the graphics
        doors onto this same function. See holographic_transfer.scatter."""
        from holographic.misc.holographic_transfer import scatter as _sc
        return _sc(np.asarray(points, float), np.asarray(values, float), tuple(shape),
                   kernel=kernel, periodic=periodic)

    def scatter_exact(self, points, values, shape, kernel="bilinear", periodic=False, bits=40):
        """`scatter`, PERMUTATION-INVARIANT: the grid is bit-identical however the points are ordered or split.

        A scatter is a REDUCE PER CELL, and `np.add.at` accumulates in point order -- so a float scatter of the same
        points in a different order gives a different grid. MEASURED, 4,000 points onto 16x16 with weights spanning
        16 orders of magnitude: the float scatter differs by 1.12e-08 under a permutation; this does not differ at
        all. With `kernel="nearest"` it is a permutation-invariant HISTOGRAM (float: 9.31e-09).
        See holographic_transfer.scatter_exact."""
        from holographic.misc.holographic_transfer import scatter_exact as _se
        return _se(np.asarray(points, float), np.asarray(values, float), tuple(shape),
                   kernel=kernel, periodic=periodic, bits=int(bits))

    def gather(self, grid, points, kernel="bilinear", periodic=False):
        """GATHER: sample a grid of any rank at continuous `points`, through the SAME kernel `scatter` deposits
        with -- which is what makes them adjoint. See holographic_transfer.gather."""
        from holographic.misc.holographic_transfer import gather as _g
        return _g(np.asarray(grid, float), np.asarray(points, float), kernel=kernel, periodic=periodic)

    # -- G1: the shader algebra doing non-graphics work (the heat propagator as a Pipeline) -----------------
    def diffusion_operator(self, shape, alpha, t, dx=1.0):
        """The heat equation's exact PERIODIC propagator as a composable shader-algebra Pipeline: `exp(-alpha|k|^2 t)`
        held once and applied many times. Bit-identical to `diffuse_spectral` (max|diff| 0.0e+00) and ~1.9x faster on
        reuse, because the transfer is composed once rather than re-exponentiated per call -- and it COMPOSES: two
        half-steps multiply into one full step, exact to 1.1e-15.

        Nothing in `Pipeline` knows what a pixel is. SCOPE, the same gate everywhere: the operator must be LINEAR
        *and* CIRCULAR. Applying this to a Neumann (edge-replicated) problem is measured 4.76e-02 wrong, and
        `diffuse_heat` keeps stepping there. See holographic_laplacian.diffusion_operator."""
        from holographic.simulation_and_physics.holographic_laplacian import diffusion_operator as _do
        return _do(tuple(shape), float(alpha), float(t), dx=float(dx))

    def diffusion_transfer(self, shape, alpha, t, dx=1.0):
        """The heat propagator's Fourier transfer, `exp(-alpha|k|^2 t)`, as a plain array. Every mode decays; the DC
        mode's transfer is exactly 1, which is why periodic diffusion conserves total heat.
        See holographic_laplacian.diffusion_transfer."""
        from holographic.simulation_and_physics.holographic_laplacian import diffusion_transfer as _dt
        return _dt(tuple(shape), float(alpha), float(t), dx=float(dx))

    # -- C1/C2: island sleep in the softbody solver, and the coordinator's colour schedule ------------------
    def softbody_sleep_tracker(self, sleep_energy=1e-8, sleep_frames=4, wake_energy=None):
        """A SleepTracker sized for a SoftBody's islands. Pass it as `body.step(..., sleep=tracker)`: islands at
        their fixed point are skipped, their state carried through BIT-IDENTICALLY, and the saving is exactly the
        awake fraction (`body.last_step_stats` counts it). A sleeping island cannot wake itself -- its velocity is
        never integrated -- so `body.wake_all()` (or an external event) must say so, which is Catto's design.
        See holographic_island.SleepTracker and holographic_softbody.SoftBody.step."""
        from holographic.simulation_and_physics.holographic_island import SleepTracker
        return SleepTracker(sleep_energy=sleep_energy, sleep_frames=sleep_frames, wake_energy=wake_energy)

    def run_waves(self, items, keys_of, worker, cache=None, coordinator=None):
        """Schedule CONFLICTING work into lock-free WAVES: two items in one wave never touch a shared resource, so
        a wave runs with no locks and no atomics, in a deterministic order. Returns (results in ITEM order, info)
        with info = {waves, wave_sizes, parallelism}. MEASURED: 2,000 transactions over 300 keys -> 24 waves, 83.3x
        lock-free parallelism, every wave conflict-free. Colouring cannot invent parallelism: if everything
        conflicts it honestly serialises. See holographic_coordinator.Coordinator.run_waves."""
        from holographic.scene_and_pipeline.holographic_coordinator import Coordinator
        own = coordinator is None
        c = Coordinator() if own else coordinator
        try:
            return c.run_waves(list(items), keys_of, worker, cache=cache)
        finally:
            if own:
                c.close()

    # -- B1: kernel fusion over the linear post-effects (postfx on the shader algebra) ----------------------
    def postfx_fuse_transfers(self, shape, steps):
        """Compose a RUN of linear, shift-invariant post-effects (denoise, sharpen) into ONE transfer -- the
        elementwise product of theirs, because diagonal operators commute and multiply. Returns None if any step is
        not fusable (`motion_blur` and `glare` clamp their edges, so they have no exact transfer). This is
        holographic_shader.Pipeline in image space. See holographic_postfx.fuse_transfers."""
        from holographic.rendering.holographic_postfx import fuse_transfers
        return fuse_transfers(tuple(shape), list(steps))

    def postfx_apply_transfer(self, img, transfer):
        """Evaluate a composed transfer: ONE FFT pair per channel, whatever the run length was. Measured on a
        3-stage run: 14.76 ms sequential vs 5.03 ms fused, max|diff| 4.44e-16. See holographic_postfx.apply_transfer."""
        from holographic.rendering.holographic_postfx import apply_transfer
        return apply_transfer(np.asarray(img, float), np.asarray(transfer))

    def postfx_fusable_runs(self, steps):
        """Split a post-chain's steps into maximal [(is_fused, [steps])] runs. MEASURED: the shipped chains have NO
        adjacent linear stages -- every blur is separated by a nonlinear tone curve -- so fusion is a capability for
        chains that HAVE such runs, not a speedup of default_chain. See holographic_postfx.fusable_runs."""
        from holographic.rendering.holographic_postfx import fusable_runs
        return [(bool(f), [(n, dict(p)) for n, p in g]) for f, g in fusable_runs(list(steps))]

    # -- A1/A2: the two correctness fixes -- partition-invariant reduction, and swept collision --------------
    def distribute_exact(self, buckets, worker, cache=None, bits=40):
        """`distribute`, but BIT-IDENTICAL under any bucketing. `worker(bucket, cache)` must return the bucket's
        CONTRIBUTIONS, not their sum -- that contract change IS the fix. The default float reduce disagrees by
        2.98e-08 between a 4-way and a 7-way split, and swapping in reduce_sum_exact does NOT repair it, because
        each worker already float-summed inside its bucket. Exactness has to reach the leaves. Returns
        (total, info) with the scale used, so the result is auditable. See holographic_distribute.distribute_exact."""
        from holographic.scene_and_pipeline.holographic_distribute import distribute_exact as _de
        return _de(list(buckets), worker, cache=cache, bits=int(bits))

    def resolve_swept_collision(self, X_prev, X, sdf_eval, radius=0.0):
        """Positional CCD for a PBD-style solver: a node that MOVED from X_prev to X must not have crossed the
        collider on the way, even if neither endpoint is inside it. Nodes whose segment does not hit are returned
        BIT-IDENTICALLY, so this is a strict addition to resolve_sdf_collision. Measured: 0.5 m/frame through a
        0.1 m wall is caught. See holographic_collide.resolve_swept_collision."""
        from holographic.simulation_and_physics.holographic_collide import resolve_swept_collision as _rs
        return _rs(X_prev, X, sdf_eval, radius=radius)

    # -- THE MACHINE MODEL: leCore's hardware units and memory tiers, named and measured --------------------
    def machine_map(self, kind=None):
        """THE SPEC SHEET: every hardware unit this engine has, as plain data -- {name, gpu_name, kind, module,
        symbol, setup, marginal, scaling, use_when, do_not_use_when, why}. `kind` filters to 'compute' or 'memory'.
        Read it before building anything that smells like a cache, a kernel, a scheduler or a lookup: the odds are
        the unit already exists and has a measured cost model. See holographic_machinemodel.machine_map."""
        from holographic.misc.holographic_machinemodel import machine_map as _mm
        return _mm(kind)

    def machine_unit(self, name):
        """One unit's full record, including the conditions under which it must NOT be used.
        See holographic_machinemodel.unit."""
        from holographic.misc.holographic_machinemodel import unit as _u
        return _u(name)

    def machine_units(self, kind=None):
        """The unit names, optionally filtered to 'compute' or 'memory'. The discovery door.
        See holographic_machinemodel.units."""
        from holographic.misc.holographic_machinemodel import units as _un
        return _un(kind)

    def machine_place(self, baseline_ns, n_calls, setup_ns, marginal_ns):
        """Should this work go on a unit, or stay on the baseline? Pure arithmetic over a measured cost model:
        {use_unit, break_even_n, unit_total_ns, baseline_total_ns, speedup}. `break_even_n` is `inf` when the
        unit's marginal cost is not below the baseline -- it can NEVER pay, and saying so is the answer. This is
        `should_jump` and `worth_factoring`, stated once. See holographic_machinemodel.place."""
        from holographic.misc.holographic_machinemodel import place as _p
        return _p(float(baseline_ns), int(n_calls), float(setup_ns), float(marginal_ns))

    def machine_place_unit(self, name, baseline_ns, n_calls, sheet=None):
        """`machine_place` on MEASURED numbers: look the unit's (setup, marginal) up in a spec sheet instead of
        supplying them. Returns the placement plus {unit, note, setup_ns, marginal_ns}. Pass a cached `sheet` from
        machine_spec_sheet() to avoid re-measuring.

        `baseline_ns` must be the cost of what the unit REPLACES, per call. Priced against a raw array read (130 ns)
        almost every unit correctly reports break_even_n = inf -- they replace an expensive evaluator, not an array
        index. If everything says never, check the denominator. See holographic_machinemodel.place_unit."""
        from holographic.misc.holographic_machinemodel import place_unit as _pu
        return _pu(str(name), float(baseline_ns), int(n_calls), sheet=sheet)

    def machine_spec_sheet(self, quick=True):
        """MEASURE the units on THIS machine, now, rather than trusting a comment: {unit: {setup_ns, marginal_ns,
        note}}. A spec sheet that cannot re-measure itself is a rumour. See holographic_machinemodel.spec_sheet."""
        from holographic.misc.holographic_machinemodel import spec_sheet as _ss
        return _ss(quick=bool(quick))

    # -- W1: COMPRESSED-DOMAIN COMPUTE -- operate on the factors, never form the field ------------------------
    def low_rank_field(self, X, rank=None, energy=0.99):
        """Factor a 2-D field into a LowRankField and operate on it WITHOUT ever materialising it: .blur(k) (a
        SEPARABLE kernel), .add(other), .scale(a), .query(i,j), .to_dense(). MEASURED (1024^2, rank 3): 171x fewer
        bytes; blur 2.53 ms / 0.049 MB vs 66.60 ms / 8.4 MB dense at 3.1e-15 error; a point query touches 72 bytes.
        You do not out-bandwidth a GPU -- you never touch the decompressed data.
        See holographic_tucker.LowRankField."""
        from holographic.caching_and_storage.holographic_tucker import LowRankField
        return LowRankField.from_dense(np.asarray(X, float), rank=rank, energy=energy)

    def worth_factoring(self, X, energy=0.99, max_error=None):
        """Would factoring this field actually save bytes? {worth_factoring, factored_bytes, dense_bytes}.

        WITHOUT `max_error` this is an ENERGY gate: it declines white noise, but it will bless a field whose
        99%-energy reconstruction is badly wrong. MEASURED on real 128x128 fields: a sphere SDF slice at 99% energy
        is rank 2 and **7.45% wrong**; a box SDF 18.19%; fbm noise passes at rank 5 with 28.54% error. An SDF that
        wrong does not sphere-trace.

        WITH `max_error` (an absolute max-abs budget) it is an ERROR gate, and that is what a consumer of the
        RECONSTRUCTION needs. At 1% of amplitude: sphere SDF rank 4 (16x fewer bytes, pays), box SDF rank 12 (5.3x),
        fbm rank 50 (1.27x, marginal), white noise rank 124 (refused).
        See holographic_tucker.LowRankField.worth_factoring."""
        from holographic.caching_and_storage.holographic_tucker import LowRankField
        ok, fb, db = LowRankField.worth_factoring(np.asarray(X, float), energy=energy, max_error=max_error)
        return {"worth_factoring": bool(ok), "factored_bytes": int(fb), "dense_bytes": int(db)}

    def rank_for_error(self, X, max_abs_error):
        """The SMALLEST rank whose truncated SVD reconstructs `X` to within `max_abs_error` in the MAX-ABS norm.
        `rank_gate`'s 99% ENERGY criterion is not this: 99% of the energy is not a small error. Measured on real
        fields, the energy rank leaves 7.45% (sphere SDF), 18.19% (box SDF) and 28.54% (fbm noise) of the amplitude
        as residual. Size on error. See holographic_tucker.rank_for_error."""
        from holographic.caching_and_storage.holographic_tucker import rank_for_error as _rfe
        r = _rfe(np.asarray(X, float), float(max_abs_error))
        return None if r is None else int(r)

    def factored_field_report(self, X, kernel, rank=None):
        """The W1 comparison, carried WITH the capability: run a separable blur dense and factored, and report
        {rank, dense_bytes, factored_bytes, byte_ratio, max_error}. Plain data, so an agent can re-run the claim
        rather than trust it. See holographic_tucker.LowRankField."""
        import numpy as _np
        from holographic.caching_and_storage.holographic_tucker import LowRankField
        X = _np.asarray(X, float)
        k = _np.asarray(kernel, float)
        f = LowRankField.from_dense(X, rank=rank)
        dense = _np.apply_along_axis(lambda c: _np.convolve(c, k, "same"), 1,
                                     _np.apply_along_axis(lambda c: _np.convolve(c, k, "same"), 0, X))
        got = f.blur(k).to_dense()
        return {"rank": f.rank, "dense_bytes": int(X.nbytes), "factored_bytes": int(f.nbytes()),
                "byte_ratio": float(X.nbytes / f.nbytes()), "max_error": float(_np.abs(got - dense).max())}

    # -- W5: HIERARCHICAL SUPERPOSITION with a mid-level cleanup (R3's third codebook consumer) --------------
    def hierarchical_pack(self, group_keys, chunk_vectors):
        """Superpose G group-keyed CHUNKS into one vector, each chunk itself a pack of its leaves. Deliberately just
        `pack`: the hierarchy is NOT in the packing (superposition is linear, so nesting alone IS the flat bundle --
        measured to 2.78e-16) but in the recall, which cleans up between levels.
        See holographic_superposed.hierarchical_pack."""
        from holographic.misc.holographic_superposed import hierarchical_pack as _hp
        return _hp(np.asarray(group_keys, float), np.asarray(chunk_vectors, float))

    def hierarchical_recall(self, S, group_key, leaf_key, chunk_codebook, item_codebook,
                            min_chunk_similarity=None):
        """Descend a hierarchical superposition with a CLEANUP at the middle level: unbind the group key, snap the
        noisy chunk to its exact pattern in `chunk_codebook` (the crosstalk reset), then unbind the leaf and snap to
        `item_codebook`. Returns {item_index, item, chunk_index, chunk_similarity, item_similarity, abstained}.

        MEASURED at D=2048, 64 groups x 8 leaves: 100% here vs 18.3% for flat_recall. Capacity is bounded by the
        WORST SINGLE LEVEL, not by the product of levels.

        `min_chunk_similarity` is the ABSTAIN GATE, and it exists because of a measured failure: if the group being
        recalled is NOT in `chunk_codebook`, the mid-level cleanup snaps to the nearest entry -- the WRONG chunk --
        and returns an item with every appearance of success. Measured: uncovered 0.036, covered 0.502; a 0.15
        threshold separates them. Default None keeps the historical behaviour exactly.
        See holographic_superposed.hierarchical_recall."""
        from holographic.misc.holographic_superposed import hierarchical_recall as _hr
        return _hr(np.asarray(S, float), np.asarray(group_key, float), np.asarray(leaf_key, float),
                   np.asarray(chunk_codebook, float), np.asarray(item_codebook, float),
                   min_chunk_similarity=min_chunk_similarity)

    def chunk_codebook_vectors(self, codebook, item_codebook, leaf_keys):
        """THE R3 BRIDGE: realize a LEARNED chunk codebook (mind.learn_chunks, R1) as the chunk vectors the
        hierarchy cleans up against. Returns (chunk_vectors, chunk_ids, leaf_lists). R1 learns WHICH chunks recur;
        R2 realizes each as a map_bind PRODUCT; this realizes each as a pack SUPERPOSITION. Same identities,
        different vectors -- that is what "one codebook family" means. See holographic_superposed."""
        from holographic.misc.holographic_superposed import chunk_codebook_vectors as _ccv
        v, ids, leaves = _ccv(codebook, np.asarray(item_codebook, float), np.asarray(leaf_keys, float))
        return v, [int(i) for i in ids], [[int(x) for x in ls] for ls in leaves]

    def chunk_coverage(self, codebook, groups, n_leaves):
        """What fraction of the groups actually used are PRESENT in the learned chunk codebook?
        {covered, total, fraction, missing}. Not bookkeeping: an uncovered group's mid-level cleanup snaps to the
        wrong chunk, so coverage is the precondition of the capacity claim, and it is set by how many merges R1 was
        allowed (measured: 60 merges covered 8 of 16 groups; 150 covered all 16).
        See holographic_superposed.chunk_coverage."""
        from holographic.misc.holographic_superposed import chunk_coverage as _cc
        return _cc(codebook, groups, int(n_leaves))

    def flat_recall(self, S, group_key, leaf_key, item_codebook):
        """The BASELINE hierarchical_recall must beat: unbind both roles and clean up ONCE at the bottom, with no
        mid-level snap. Shipped beside its competitor so the comparison can be re-run, not taken on trust.
        See holographic_superposed.flat_recall."""
        from holographic.misc.holographic_superposed import flat_recall as _fr
        return _fr(np.asarray(S, float), np.asarray(group_key, float), np.asarray(leaf_key, float),
                   np.asarray(item_codebook, float))

    # -- LEARNED CHUNK CODEBOOKS: iterated pair promotion (backlog R1; the ONE codebook family, R3) -----------
    # Every method here takes and returns PLAIN DATA (lists of ints, a dict codebook), so a codebook crosses an
    # HTTP boundary and a long-lived service hands the SAME one to the resonator (R2), a chunked store (W5) and
    # the edit codec (DL8). A live ChunkCodebook object is available via chunk_codebook() for in-process use.
    def learn_chunks(self, stream, max_merges=200, min_count=2):
        """Learn a CHUNK CODEBOOK from a symbol stream by iterated pair promotion (BPE -- Gage 1994; Sennrich et
        al. 2016), where the merged chunks are factoring and storage codebooks rather than tokenizer vocabulary.
        Returns plain data {merges, depth}. Deterministic: ties on count break on the pair, never on dict order.
        KEPT NEGATIVE: this is NOT a byte compressor -- zlib beats it on the same stream (measured 1,820 vs 3,578
        bytes). Its value is the codebook it leaves behind. See holographic_chunkcodebook.learn_chunks."""
        from holographic.agents_and_reasoning.holographic_chunkcodebook import learn_chunks as _lc
        return _lc(list(stream), max_merges=int(max_merges), min_count=int(min_count)).to_dict()

    def chunk_codebook(self, codebook):
        """Rehydrate a plain-data codebook into a live ChunkCodebook (with .encode/.decode/.stats). The in-process
        twin of the plain-data methods. See holographic_chunkcodebook.ChunkCodebook."""
        from holographic.agents_and_reasoning.holographic_chunkcodebook import ChunkCodebook
        return ChunkCodebook.from_dict(codebook)

    def chunk_encode(self, stream, codebook):
        """Tokenize `stream` against a learned codebook by replaying its merges in learning order. Returns a token
        list. See holographic_chunkcodebook.ChunkCodebook.encode."""
        return [int(t) for t in self.chunk_codebook(codebook).encode(list(stream))]

    def chunk_decode(self, tokens, codebook):
        """Expand chunk tokens back to base symbols. LOSSLESS: chunk_decode(chunk_encode(s)) == s exactly.
        See holographic_chunkcodebook.ChunkCodebook.decode."""
        return [int(s) for s in self.chunk_codebook(codebook).decode(list(tokens))]

    def chunk_stats(self, stream, codebook):
        """{n_symbols, n_tokens, token_ratio, mean_depth, max_depth, n_merges, covered}. `token_ratio` is a TOKEN
        count ratio, not byte compression. See holographic_chunkcodebook.ChunkCodebook.stats."""
        return self.chunk_codebook(codebook).stats(list(stream))

    def structure_score(self, stream, max_merges=200, min_count=2):
        """Does this stream have REUSABLE STRUCTURE? The mean chunk depth of its own learned tokenization: 1.0 =
        nothing merged. Measured 4.31 on a workflow stream, 1.34 on a uniform control. This is the gate on the
        recursion dividend -- R2's recursive factoring, W5's hierarchical superposition and DL8's edit codec all
        pay off only when a stream is made of reused chunks. See holographic_chunkcodebook.structure_score."""
        from holographic.agents_and_reasoning.holographic_chunkcodebook import structure_score as _ss
        return float(_ss(list(stream), max_merges=int(max_merges), min_count=int(min_count)))

    def chunk_byte_report(self, stream, codebook):
        """The codec comparison, carried WITH the capability because the token ratio invites a claim the bytes do
        not support: {raw, zlib_raw, bpe_bytes, zlib_bpe_bytes, beats_zlib}. beats_zlib is False by design.
        See holographic_chunkcodebook.byte_report."""
        from holographic.agents_and_reasoning.holographic_chunkcodebook import byte_report
        return byte_report(list(stream), self.chunk_codebook(codebook))

    # -- PHYSICS EVENT CODEC: a trace as base + interruptions (Box3D B8 / backlog X7) -------------------------
    def record_physics_trace(self, n=16, frames=600, seed=0, dt=1.0 / 120.0, gravity=-9.81,
                             restitution=0.8, box=2.0, speed=2.0):
        """Record the reference bouncing-box trace: `n` bodies under gravity in a restitution box. Returns
        (trace (frames,n,6), EventTrace). Deterministic -- fixed integrator, fixed clamp, fixed event order.
        See holographic_eventcodec.bouncing_box_trace."""
        from holographic.simulation_and_physics.holographic_eventcodec import bouncing_box_trace
        return bouncing_box_trace(n=n, frames=frames, seed=seed, dt=dt, gravity=gravity,
                                  restitution=restitution, box=box, speed=speed)

    def replay_physics_trace(self, ev, dt=None, gravity=None, box=None):
        """Regenerate a full trace from base + events, BIT-IDENTICALLY (measured max|diff| exactly 0.0). Between
        events physics is a deterministic function of the state, so the states were never data. Bit-exactness is
        required, not a nicety: a contact sequence is a chaotic map. See holographic_eventcodec.replay_bouncing_box."""
        from holographic.simulation_and_physics.holographic_eventcodec import replay_bouncing_box
        return replay_bouncing_box(ev, dt=dt, gravity=gravity, box=box)   # None => the trace's own parameters

    def physics_compression_report(self, trace, ev):
        """The event codec's size AND every baseline it claims to beat, computed together so an agent can re-run
        the comparison instead of trusting the number: {raw, zlib_raw, zlib_frame_deltas, deltachain, event_codec,
        ratio_vs_frame_deltas, events, rows}. MEASURED: 6,360 bytes vs zlib(frame deltas) 87,057 -- 13.7x, lossless.
        See holographic_eventcodec.compression_report."""
        from holographic.simulation_and_physics.holographic_eventcodec import compression_report
        return compression_report(trace, ev)

    # -- MODAL JUMP: a linear island advanced in closed form within a contact mode (Box3D B1 / backlog X1) ----
    def soft_chain_matrices(self, n_bodies, hertz=15.0, zeta=0.7, mass=1.0, gravity=-9.81,
                            dt=1.0 / 60.0, substeps=64):
        """Build an anchored soft-constraint chain's implicit-Euler substep as the affine map `s <- A s + b`
        (s = [positions; velocities]). Parameterized in Catto's units (hertz, zeta), like soft_relaxation.
        Returns (A, b, h). The reference scene for the modal jump. See holographic_modal.soft_chain_matrices."""
        from holographic.simulation_and_physics.holographic_modal import soft_chain_matrices as _scm
        A, b, h = _scm(n_bodies, hertz=hertz, zeta=zeta, mass=mass, gravity=gravity, dt=dt, substeps=substeps)
        return A, b, h

    def affine_jump(self, state, A, b, k):
        """THE MODAL JUMP, stateless: advance the affine recurrence `s <- A s + b` by k substeps in ONE
        eigendecomposition. Within a contact mode a soft-constraint island IS this recurrence, so 3,840 substeps
        cost one solve -- and so do 38,400. Measured: matches the substepped reference to 2.5e-12 on a 12-body
        chain. Raises if the operator is DEFECTIVE (a free-body Jordan block has no eigenbasis: step it instead).
        The stateless one-shot twin of modal_solver, so an agent can call it over HTTP.
        See holographic_iterate.affine_step_k."""
        from holographic.misc.holographic_iterate import affine_step_k
        return affine_step_k(np.asarray(state, float), np.asarray(A, float), np.asarray(b, float), int(k))

    def affine_limit(self, A, b):
        """The k -> infinity fixed point of `s <- A s + b` (a settled island), computed per-mode so it can refuse
        honestly: raises if any |eigenvalue| >= 1, because a marginal or divergent island never settles and a
        number there would be a fabrication. The dense twin of settle_island. See holographic_iterate.affine_limit."""
        from holographic.misc.holographic_iterate import affine_limit as _al
        return _al(np.asarray(A, float), np.asarray(b, float))

    def should_jump(self, dim, k):
        """Does jumping k substeps of a dim-dimensional island beat stepping them? True when k >= 20*dim --
        MEASURED, since an eigendecomposition is O(dim^3) and a substep is O(dim^2). The descriptor's jump-vs-step
        gate. See holographic_modal.should_jump."""
        from holographic.simulation_and_physics.holographic_modal import should_jump as _sj
        return bool(_sj(int(dim), int(k)))

    def modal_solver(self, state, breakeven=20.0, cache_modes=True):
        """A stateful ModalSolver: set_mode(key, A, b) on every contact-mode switch, advance(k) to jump within a
        mode, report() for the economics (switches, substeps_per_switch, fallbacks). Re-diagonalizes only at
        switches and caches each mode's factorization, so a cycling machine pays per mode once, ever. Factorizes
        LAZILY -- a mode below the break-even is stepped and never pays for an eigendecomposition it declines.
        A live handle: use affine_jump for the stateless/HTTP path. See holographic_modal.ModalSolver."""
        from holographic.simulation_and_physics.holographic_modal import ModalSolver
        return ModalSolver(state, breakeven=breakeven, cache_modes=cache_modes)

    def soft_chain_bank(self, n_bodies, hertz, zeta, mass=1.0, gravity=-9.81, dt=1.0 / 60.0, substeps=64):
        """A TUNING BANK (X8): M variants of one soft-constraint chain differing in stiffness/damping (`hertz` and
        `zeta` scalar or length-M). Returns (As, bs, h) for advance_bank. NOT a superposition -- a trajectory is
        nonlinear in the operator, so stiffness variants cannot be summed into one vector at any dimension; they are
        batched as arrays, which is exact and has no capacity budget. See holographic_modal.soft_chain_bank."""
        from holographic.simulation_and_physics.holographic_modal import soft_chain_bank as _scb
        As, bs, h = _scb(n_bodies, hertz, zeta, mass=mass, gravity=gravity, dt=dt, substeps=substeps)
        return As, bs, h

    def advance_bank(self, S0, As, bs, k):
        """Advance M island variants k substeps in ONE batched closed form: (M,2n) -> (M,2n). Measured, M=32 x 1,920
        substeps: 13.20 ms substepped vs 3.10 ms here (4.3x), max|diff| 1.86e-12, horizon free. Raises naming the
        variant if any member is defective. See holographic_modal.advance_bank."""
        from holographic.simulation_and_physics.holographic_modal import advance_bank as _ab
        return _ab(np.asarray(S0, float), np.asarray(As, float), np.asarray(bs, float), int(k))

    def blend_forcings(self, s0, A, forcings, weights, k):
        """The one thing that DOES superpose exactly: variants differing only in their FORCING. `s_k = A^k s0 +
        G(A,k) b` is linear in b, so a weighted blend of forcings is the blend of trajectories (measured 1.11e-16).
        One eigendecomposition serves every forcing variant and every blend -- gravity, wind, a load dial. Blending
        OPERATORS instead is nonsense (measured 2.94e-01). See holographic_modal.blend_forcings."""
        from holographic.simulation_and_physics.holographic_modal import blend_forcings as _bf
        return _bf(np.asarray(s0, float), np.asarray(A, float), np.asarray(forcings, float),
                   np.asarray(weights, float), int(k))

    def escalation_plan(self, dim, k, energy=None, sleep_energy=0.0, diagonalizable=True, breakeven=20.0):
        """THE ESCALATION LADDER (X11): pick {sleep | jump | substep} for one island, per frame. Catto's "4
        substeps" dial and leCore's closed form are two ends of ONE axis and the descriptor chooses the rung --
        sleep if the island is at its fixed point, jump if it is linear and k amortizes an O(dim^3)
        eigendecomposition, substep otherwise (churning contacts, small k, or a defective operator with no
        eigenbasis). Returns {rung, why, substeps}: a pure decision, inspectable before anything runs, like
        plan_render. See holographic_modal.escalation_plan."""
        from holographic.simulation_and_physics.holographic_modal import escalation_plan as _ep
        return _ep(int(dim), int(k), energy=energy, sleep_energy=sleep_energy,
                   diagonalizable=diagonalizable, breakeven=breakeven)

    def mode_key(self, active_constraints):
        """A deterministic signature for a contact mode (the active-constraint set), so the same mode reached by
        different orders is recognised as the same mode. See holographic_modal.mode_key."""
        from holographic.simulation_and_physics.holographic_modal import mode_key as _mk
        return _mk(active_constraints)

    # -- ISLANDS: connected components of a constraint graph + the sleep probe (Box3D B3 / backlog X3) --------
    def islands(self, n_nodes, edges):
        """Partition n_nodes into ISLANDS -- the connected components of a constraint graph, under an undirected
        list of (u, v) edges. Returns sorted index lists ordered by smallest member (deterministic, independent of
        edge order). An island is a physics island, a mesh shell, a farm bucket, a DDM subdomain: one flood fill.
        See holographic_island.connected_components."""
        from holographic.simulation_and_physics.holographic_island import connected_components as _cc
        return _cc(n_nodes, edges)

    # -- FAT-MARGIN CACHING for a drifting query (Box3D lesson / backlog X9) ---------------------------------
    def margin_cache(self, builder, margin, metric=None):
        """Cache a baked result over an ENLARGED region around a DRIFTING query (a camera, a cursor, an agent, a
        recall neighbourhood): every query within `margin` of the bake's centre is served from it. Catto's enlarged
        AABB, generalized past physics. `.get(point) -> (value, hit)`, `.stats()` counts the trade. Measured on a
        unit-step 2-D walk: exact-key caching rebuilds 400/400; margin 6.0 rebuilds 20 at 95% hits.
        See holographic_cachehome.MarginCache."""
        from holographic.caching_and_storage.holographic_cachehome import MarginCache
        return MarginCache(builder, margin, metric=metric)

    def drift_scale(self, queries, metric=None):
        """The `variation` probe pointed at a QUERY STREAM instead of at data: the mean step between consecutive
        queries. This is the statistic that sets a fat margin. See holographic_cachehome.drift_scale."""
        from holographic.caching_and_storage.holographic_cachehome import drift_scale as _ds
        return _ds(queries, metric=metric)

    def replay_margin(self, queries, margin, metric=None):
        """Replay a query stream against a margin and count what it would cost: {hits, rebuilds, hit_rate, margin}.
        No baking, so it is cheap enough to sweep. See holographic_cachehome.replay_margin."""
        from holographic.caching_and_storage.holographic_cachehome import replay_margin as _rm
        return _rm(queries, margin, metric=metric)

    def replay_margin_error(self, queries, values, margin, metric=None):
        """Replay a query stream against a margin and measure what the cache would have SERVED against what was TRUE:
        {hits, rebuilds, hit_rate, margin, max_error, mean_error}. `replay_margin` counts hits; a hit serves a STALE
        value, and hit rate says nothing about how stale. On a rendered frame the max error saturates at the FIRST
        reuse (0.5864) while the mean creeps 0.0001 -> 0.0051. See holographic_cachehome.replay_margin_error."""
        from holographic.caching_and_storage.holographic_cachehome import replay_margin_error as _rme
        return _rme(queries, values, float(margin), metric=metric)

    def suggest_margin_for_error(self, queries, values, max_mean_error, max_abs_error=None, metric=None):
        """THE GATE for any fat-margin cache: the LARGEST margin whose replayed error stays inside the budget.
        `max_abs_error` bounds the WORST served result, and you usually want it -- measured, a value that jumps 0->1
        passes a mean-only budget at margin 0.1929 and serves a completely wrong answer (max error 1.00). The
        max-error bound stops at 0.094558, and 0.095158 is already catastrophic: the admissible margin is a CLIFF,
        not a slope. Size on the error you cannot tolerate. See holographic_cachehome.suggest_margin_for_error."""
        from holographic.caching_and_storage.holographic_cachehome import suggest_margin_for_error as _sme
        return float(_sme(queries, values, float(max_mean_error), max_abs_error=max_abs_error, metric=metric))

    def suggest_margin(self, queries, target_hit_rate=0.95, metric=None):
        """The SMALLEST margin whose replayed hit rate on this query stream meets the target. Empirical by design:
        a random walk's exit time scales like (R/sigma)^2, but measured rebuilds sit ~1.8x off that, so replaying
        the stream you actually have beats a fitted law. See holographic_cachehome.suggest_margin."""
        from holographic.caching_and_storage.holographic_cachehome import suggest_margin as _sm
        return float(_sm(queries, target_hit_rate=target_hit_rate, metric=metric))

    def conflict_graph(self, item_keys):
        """Build the CONFLICT GRAPH of a batch of tasks -- `item_keys[i]` is the set of resources task i touches,
        and two tasks are adjacent iff they share one. Returns (n_items, edges) for graph_coloring / color_waves.
        Built key-first, so the cost is the sum of squared key degrees, not O(n^2). See holographic_island."""
        from holographic.simulation_and_physics.holographic_island import conflict_graph as _cg
        n, e = _cg([set(k) for k in item_keys])
        return n, [list(x) for x in e]

    def graph_coloring(self, n_nodes, edges):
        """Greedy graph colouring: each node takes the smallest colour no neighbour uses. DETERMINISTIC by
        construction (ascending visit order), which is the point -- Catto's parallel solver earns its
        cross-platform determinism exactly this way. See holographic_island.graph_coloring."""
        from holographic.simulation_and_physics.holographic_island import graph_coloring as _gc
        return _gc(int(n_nodes), edges)

    def color_waves(self, n_nodes, edges):
        """Partition conflicting tasks into WAVES that touch disjoint resources: each wave runs fully parallel with
        no locks and no atomics, in a reproducible order. Measured: 2,000 transactions over 300 keys colour into 24
        waves -- 83x lock-free parallelism. See holographic_island.color_waves."""
        from holographic.simulation_and_physics.holographic_island import color_waves as _cw
        return _cw(int(n_nodes), edges)

    def plan_write_waves(self, batch_keys):
        """Schedule database write batches into key-disjoint waves (X10): the single-writer lock serialises writers
        because two might touch the same row; colouring PROVES when they cannot. Deterministic schedule.
        See holographic_querylock.plan_write_waves."""
        from holographic.agents_and_reasoning.holographic_querylock import plan_write_waves as _pw
        return _pw(batch_keys)

    def scan_exact(self, x, bits=40):
        """The PREFIX SUM, bit-identical however the work is blocked. A blocked FLOAT scan -- what every parallel
        scan actually is -- gives a different answer per block count: measured, 4-block vs 7-block disagree by
        1.14e-12 on uniform data, 3.87e-07 across 16 orders of magnitude, and 92.0 (9.2e-15 relative) on
        [1e16, 1, -1e16] repeated. Quantize once against a global scale and scan in int64, where addition IS
        associative. KEPT NEGATIVE: this is not more ACCURATE than np.cumsum -- it is more REPRODUCIBLE. A
        sequential cumsum beats it on precision every time; it just cannot run on eight blocks and give the same
        bits. See holographic_distribute.scan_exact."""
        from holographic.scene_and_pipeline.holographic_distribute import scan_exact as _se
        return _se(np.asarray(x, float), bits=int(bits))

    def scan_exact_blocked(self, x, n_blocks, bits=40):
        """`scan_exact` computed the way a GPU or a farm would -- a cumsum per block plus the exclusive prefix of
        the block sums -- and BIT-IDENTICAL to it for every block count from 1 to N, because the carries are int64.
        This is the function that makes the claim testable. See holographic_distribute.scan_exact_blocked."""
        from holographic.scene_and_pipeline.holographic_distribute import scan_exact_blocked as _seb
        return _seb(np.asarray(x, float), int(n_blocks), bits=int(bits))

    def reduce_sum_exact_partitioned(self, buckets, bits=40):
        """Sum a list of BUCKETS of contributions, bit-identically under ANY bucketing -- 4-way, 7-way, or one
        bucket per contribution. This is the invariance `reduce_sum_exact` does NOT have once a farm float-sums
        inside its buckets (measured: exact on the raw contributions, NOT on the bucket sums). One global scale from
        the global peak and count (max and len are partition-invariant), then integer accumulators that merge in any
        order. Determinism that survives RE-PARTITIONING a running farm. See holographic_distribute."""
        from holographic.scene_and_pipeline.holographic_distribute import reduce_sum_exact_partitioned as _rp
        return _rp(buckets, bits=bits)

    def island_energy(self, positions, velocities=None, rest=None):
        """The SLEEP SENSOR for one island: a monotone 'how active is this' scalar (kinetic + displacement from
        rest). Feed it to island_sleep_tracker to decide awake vs asleep. See holographic_island.island_energy."""
        from holographic.simulation_and_physics.holographic_island import island_energy as _ie
        return _ie(positions, velocities=velocities, rest=rest)

    def island_sleep_tracker(self, sleep_energy=1e-8, sleep_frames=4, wake_energy=None):
        """A per-island awake/asleep tracker with HYSTERESIS: an island sleeps after `sleep_frames` consecutive
        frames under `sleep_energy`, and wakes the instant it exceeds `wake_energy` (default 4x). Two thresholds,
        because one flickers on floating-point noise -- measured. See holographic_island.SleepTracker."""
        from holographic.simulation_and_physics.holographic_island import SleepTracker
        return SleepTracker(sleep_energy=sleep_energy, sleep_frames=sleep_frames, wake_energy=wake_energy)

    def step_islands(self, state, islands, step, tracker=None, energies=None):
        """Advance ONLY the awake islands one frame: (new_state, awake_ids, asleep_ids). A sleeping island's rows
        are carried through BIT-IDENTICALLY (skipping must not perturb). With tracker=None everything is awake --
        the old behaviour exactly. See holographic_island.step_islands."""
        from holographic.simulation_and_physics.holographic_island import step_islands as _si
        return _si(state, islands, step, tracker=tracker, energies=energies)

    def settle_island(self, state, U, tol=1e-6):
        """SLEEP IS THE CLOSED FORM: send an island straight to the fixed point of `x <- bind(U, x)` in ONE
        evaluation instead of stepping until it falls asleep -- this is iterate.limit() wearing the physics
        costume. Measured: the fixed point is NOT rest in general; modes with |eigenvalue| ~ 1 persist, so a
        diffusive island settles to its MEAN. Only a strictly contractive operator settles to zero.
        See holographic_island.settle_island."""
        from holographic.simulation_and_physics.holographic_island import settle_island as _st
        return _st(state, U, tol=tol)

    def soft_relaxation(self, hertz, zeta, dt):
        """The per-substep relaxation factor of a SOFT constraint of stiffness `hertz` (natural frequency) and
        damping ratio `zeta` (1.0 = critically damped), stepped at `dt` -- Catto's Soft Step dial, in physical
        units. Returns a float in [0, 1]; hertz=inf is the hard projection exactly. Feed it to
        project_onto_constraints(omega=...), or just pass stiffness=(hertz, zeta) there. Use it to pick a
        constraint's stiffness once and have it mean the same thing at any substep count -- a hand-tuned omega
        does not. See holographic_denoise.soft_relaxation."""
        from holographic.rendering.holographic_denoise import soft_relaxation as _sr
        return _sr(hertz, zeta, dt)

    def project_onto_constraints(self, x, projections, iters=30, tol=None, omega=1.0,
                                 stiffness=None, dt=None, return_residual=False):
        """Satisfy a set of constraints on a vector by ITERATED PROJECTION -- sweep a list of projections
        (each a callable x->x' snapping x onto one constraint set / manifold) until they jointly hold. This is
        the one engine under three faculties the mind grew separately (Macklin's observation -- the object he
        builds in position-based dynamics): the SBC resonator is alternating projection onto factor codebooks,
        `denoise(method='pnp')` is alternating projection onto data-fidelity and the signal manifold (it now
        literally calls this), and a constraint sweep is PBD. With `omega`<1 the update is under-relaxed (PBD's
        stability trick); onto convex sets this is POCS and converges to their intersection.

        SOFT CONSTRAINTS (X2): pass `stiffness=(hertz, zeta)` and the substep `dt` instead of hand-tuning omega,
        and the constraint is specified in PHYSICAL units -- natural frequency in hertz, damping ratio zeta
        (1.0 = critically damped). Catto's Soft Step parameterization: the same (hertz, zeta) means the same
        physics at ANY substep count, where the same omega does not. `stiffness=(inf, zeta)` is the hard
        projection exactly, so every existing caller is unchanged. This is what gives PBD, FABRIK/IK, the
        resonator and the PnP denoise loop Catto-style softness from ONE dial. Returns
        (x, n_sweeps, converged). `return_residual=True` appends the final constraint residual
        (x, n_sweeps, converged, residual) = max_j ||proj_j(x)-x|| at the answer -- Macklin's XPBD discipline: how
        far the solve still sits from satisfying the constraints (the graded signal a converged boolean loses, and
        the analogue of solve_ik_limited's reach_error). Default off, so every existing 3-tuple caller is unchanged.
        Deterministic given deterministic projections. Useful directly for enforcing arbitrary constraints on a
        hypervector (be a valid code AND agree with a partial observation, say)."""
        from holographic.rendering.holographic_denoise import project_onto_constraints as _poc
        return _poc(np.asarray(x, float), list(projections), iters=iters, tol=tol, omega=omega,
                    stiffness=stiffness, dt=dt, return_residual=return_residual)

    def restore(self, y, mask=None, samples=None, forward=None, adjoint=None,
                mu=0.5, steps=30, sigma=None, rank=8):
        """Restore a degraded measurement y = A(clean) + noise by Plug-and-Play / RED -- the inverse-problem
        faculty (Milanfar's thesis: a denoiser IS a map of the signal manifold, so it is the prior in ANY
        inverse problem -- deblur, super-resolve, and especially INPAINT). The common case is inpainting an
        ERASED signal (Ozcan's degraded archive plate): pass `mask` (1 where observed, 0 where erased) and the
        forward operator A and its transpose are filled in for you -- a diagonal mask is its own transpose. For
        other operators pass `forward`/`adjoint`. The prior is THIS mind's adaptive manifold denoiser fit from
        `samples` (clean rows -- e.g. an archive's own gallery), which estimates the noise level itself, so it
        does not over-smooth at low noise.

        This is addendum B7 wired as a LOOP, not a one-shot: it delegates to `denoise(method='pnp')`, iterating
        data-fidelity <-> denoise so the erased entries are filled by the manifold while the observed ones are
        held to the measurement. That iteration is exactly why it beats a single denoise on erasure -- the
        one-shot projection of the masked signal is dragged toward zero by the missing values. Returns the
        restored vector."""
        if forward is None or adjoint is None:
            if mask is None:
                raise ValueError("restore needs a `mask` (for inpainting) or explicit forward/adjoint operators")
            mvec = np.asarray(mask, float)
            forward = adjoint = (lambda x, _m=mvec: _m * x)   # a diagonal mask is its own transpose (A == A^T)
        return self.denoise(np.asarray(y, float), method="pnp", samples=samples,
                            forward=forward, adjoint=adjoint, mu=mu, pnp_steps=steps, sigma=sigma, rank=rank)

    def fit_function(self, X, y, n_grid=24, bandwidth=8.0, ridge=1e-2):
        """Fit an interpretable function y ~ F(X) as a single-layer Kolmogorov-Arnold readout on this
        mind's encoders (holographic_kan): F(x) = sum_j psi_j(x_j), each per-feature psi_j a sum of
        adaptive-grid RBF bumps, all coefficients fit by ONE deterministic ridge least-squares solve
        (no backprop). The fitted model exposes .predict(X) and .feature_function(j, ts) -- the plottable
        learned univariate part for feature j, KAN's interpretability pitch.

        X is (N, n_features) (a 1-D array is taken as one feature); y is (N,). Returns the fitted
        HolographicKAN, built at this mind's seed so it stays seed-reproducible like the rest.

        KEPT NEGATIVE: a single-layer additive KAN cannot represent feature INTERACTIONS (e.g. x1*x2) --
        it is additive by construction. That needs a second layer or explicit interaction features; the
        boundary is shown, not hidden."""
        from holographic.agents_and_reasoning.holographic_kan import HolographicKAN
        X = np.asarray(X, float)
        if X.ndim == 1:                                # a lone feature vector -> a single column
            X = X.reshape(-1, 1)
        model = HolographicKAN(X.shape[1], dim=min(self.dim, 512), n_grid=n_grid,
                               bandwidth=bandwidth, seed=self.seed, ridge=ridge)
        return model.fit(X, np.asarray(y, float))

    # ---- the EXPLICIT 3-D GEOMETRY faculties (forward DCC backlog, FWD-1 / FWD-2) ----------------
    # The first explicit-mesh layer on the mind: a polygon mesh kernel (half-edge adjacency, Euler
    # invariants, normals, OBJ/buffer export) and the glTF (.glb) boundary that hands a scene to a
    # three.js front end. These are explicit-geometry I/O, NOT VSA hypervector ops -- a mesh is integer
    # connectivity over float positions, not a hypervector, and this layer does not pretend otherwise.
    # The bridge to the native VSA reps (mesh <-> SDF <-> splat, a mesh AS a StructureRecipe) is a later
    # item (FWD-11 / ARCH); this is the honest explicit substrate those bridges will connect to.

    def mesh_box(self, width=1.0, height=1.0, depth=1.0, center=(0.0, 0.0, 0.0)):
        """An axis-aligned box as a quad Mesh (holographic_mesh, FWD-1): the canonical first explicit mesh
        -- 8 vertices, 6 quad faces, V - E + F = 2 (a genus-0 closed surface). The returned Mesh carries
        the half-edge adjacency, Euler/manifold checks, normals, triangulation and OBJ/buffer export."""
        from holographic.mesh_and_geometry.holographic_mesh import box
        return box(width, height, depth, center)

    def mesh_tetrahedron(self, scale=1.0, center=(0.0, 0.0, 0.0)):
        """A regular tetrahedron as 4 triangles (holographic_mesh, FWD-1) -- the smallest closed manifold,
        a second primitive proving the Euler machinery is not box-specific."""
        from holographic.mesh_and_geometry.holographic_mesh import tetrahedron
        return tetrahedron(scale, center)

    def mesh_grid(self, nx=4, ny=4, width=1.0, height=1.0, center=(0.0, 0.0, 0.0)):
        """A flat subdivided plane as an (nx by ny) quad grid (holographic_mesh, FWD-1): an OPEN mesh
        (chi = 1, has a boundary) -- the test surface for boundary handling and bigger builds."""
        from holographic.mesh_and_geometry.holographic_mesh import grid
        return grid(nx, ny, width, height, center)

    def mesh_euler(self, mesh):
        """The combinatorial well-formedness signature of a Mesh (holographic_mesh, FWD-1): vertices,
        edges, faces, the Euler characteristic chi = V - E + F, whether the surface is closed and manifold,
        and -- for a closed surface -- the genus from chi = 2 - 2g. The exact-integer health check the
        explicit-geometry edit operators (FWD-7) will be required to preserve."""
        return {
            "vertices": mesh.n_vertices,
            "edges": mesh.n_edges,
            "faces": mesh.n_faces,
            "characteristic": mesh.euler_characteristic(),
            "closed": mesh.is_closed(),
            "manifold": mesh.is_manifold(),
            "genus": mesh.genus(),
        }

    def mesh_to_gltf(self, mesh, base_colour=(0.8, 0.8, 0.8, 1.0), material=None):
        """Serialise a Mesh to single-file binary glTF (.glb) bytes (holographic_gltf, FWD-2): the boundary
        a three.js GLTFLoader consumes. POSITION/NORMAL/TEXCOORD_0/COLOR_0 attributes plus a triangle index
        buffer, with the glTF-required position bounds. `material` is an optional PBRMaterial (full
        metallic-roughness + emissive factors); without it, a default base-colour PBR material is written.
        Deterministic -- the same mesh yields byte-identical output."""
        from holographic.io_and_interop.holographic_gltf import mesh_to_glb
        return mesh_to_glb(mesh, base_colour=base_colour, material=material)

    def pbr_material(self, name="material", base_color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.8,
                     emissive=(0.0, 0.0, 0.0)):
        """A standard glTF 2.0 metallic-roughness PBR material (the ISO factor model: base colour RGBA, metallic,
        roughness, emissive RGB) -- the representation every export/import maps through. See
        holographic_materialio.PBRMaterial."""
        from holographic.materials_and_texture.holographic_materialio import PBRMaterial
        return PBRMaterial(name=name, base_color=base_color, metallic=metallic, roughness=roughness,
                           emissive=emissive)

    def material_to_mtl(self, material):
        """A PBRMaterial as an MTL block (the OBJ companion): modern PBR keywords (Pr/Pm/Ke) + legacy Kd/d/Ns so
        it round-trips with PBR-aware tools and still opens in old viewers. See holographic_materialio."""
        return material.to_mtl()

    def materials_from_mtl(self, text):
        """Parse an MTL file's text into a list of PBRMaterial. See holographic_materialio.materials_from_mtl."""
        from holographic.materials_and_texture.holographic_materialio import materials_from_mtl
        return materials_from_mtl(text)

    def material_to_vsa_record(self, material, scalar_encoder):
        """Carry a PBRMaterial as ONE hypervector (each factor scalar-encoded, bound to its channel role, then
        bundled) -- so a material transmits, composes, and BLENDS with the engine's bind/bundle algebra, like a
        splat scene or a typed record. Round-trips to the encoder's resolution (crosstalk-limited; use the exact
        MTL/glTF path for lossless). See holographic_materialio.PBRMaterial.to_vsa_record."""
        return material.to_vsa_record(scalar_encoder)

    def material_from_vsa_record(self, record, scalar_encoder, name="material"):
        """Recover a PBRMaterial from its hypervector record (unbind each channel role, decode the scalar).
        See holographic_materialio.PBRMaterial.from_vsa_record."""
        from holographic.materials_and_texture.holographic_materialio import PBRMaterial
        return PBRMaterial.from_vsa_record(record, scalar_encoder, name=name)

    def mesh_from_gltf(self, data):
        """Parse binary glTF (.glb) bytes back into a Mesh (holographic_gltf, FWD-2): the inverse of
        `mesh_to_gltf`. Faces return as triangles (glTF's buffer form); positions, normals and uvs are
        recovered. The offline round-trip that proves the boundary is lossless before three.js ever sees it."""
        from holographic.io_and_interop.holographic_gltf import glb_to_mesh
        return glb_to_mesh(data)

    # ---- FWD-7: local Euler EDIT operators -- the rewrites that let the modeler CHANGE a mesh ---------
    # The four workhorses every remesher/decimator decomposes into, on the shipped half-edge kernel. Each
    # returns a NEW Mesh, keeps the surface a valid manifold, and bookkeeps the Euler characteristic exactly;
    # the make/kill pairs (flip's self-inverse, split/collapse) give exact do-then-undo round-trips. Higher
    # ops (subdivide, bevel, decimate) are sequences of these and are later items.

    def mesh_flip_edge(self, mesh, a, b):
        """Rotate the edge shared by two triangles (holographic_eulerops, FWD-7): remove edge a-b, add edge
        c-d between the opposite apexes. V/E/F (hence chi) unchanged -- the Delaunay-remeshing primitive.
        Refuses if c-d already exists (would be non-manifold). Returns a new Mesh."""
        from holographic.misc.holographic_eulerops import flip_edge
        return flip_edge(mesh, a, b)

    def mesh_split_edge(self, mesh, a, b):
        """Insert a midpoint vertex on edge {a,b}, splitting the incident triangle(s) (holographic_eulerops,
        FWD-7): the refinement primitive. V+1, chi unchanged. Returns (new_mesh, m) where m is the new vertex
        index; collapse_edge(new_mesh, keep=a, remove=m) is its exact inverse."""
        from holographic.misc.holographic_eulerops import split_edge
        return split_edge(mesh, a, b)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p05_explain_code", "_UnifiedPart05")
    print("holographic_unified_p05_explain_code selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
