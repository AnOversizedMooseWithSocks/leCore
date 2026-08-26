"""Part 02 of UnifiedMind's faculty surface -- 71 methods, fit_deterministic .. generate.

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


class _UnifiedPart02:

    def fit_deterministic(self, data, coarse=64, keep_frac=0.25, refine_steps=12, tie_tol=0.02, seed=0):
        """Recover the deterministic GENERATOR that best explains a 1-D `data` signal (the inverse of the ladder):
        SNAP the data against a baked generator bank (sine/chirp/gauss/sawtooth), then REFINE the winning family's
        params. Returns {family, params, correlation, residual_frac, ties, verdict}. Q8: the snap compares
        BAND-LIMITED signatures, so families differing only above the coarse rate TIE honestly rather than one
        winning on aliased detail. REFUSES (family=None) when no generator beats the noise -- 'no deterministic
        structure' is a result. If it fits, store (family, params) -- bytes, not samples. See
        holographic_fitgen.fit_deterministic."""
        from holographic.agents_and_reasoning.holographic_fitgen import fit_deterministic
        return fit_deterministic(data, coarse=coarse, keep_frac=keep_frac, refine_steps=refine_steps,
                                 tie_tol=tie_tol, seed=seed)

    def analyze_axes(self, data, coords=None, categorical=None):
        """Which axis of a tensor is the INDEX (carrier) vs the PAYLOAD (content)?
        (holographic_axisrole). Measures, per axis, its marginal information (how
        boring/regular it is) and its content coupling (how much the content varies
        along it), then recommends INDEX (keep it as an enumeration/carrier -- the
        cheap, comparability-preserving default) or BIND (fold its value into the
        content vector -- only when the axis is informative AND its conjunction with
        content is the meaningful unit). This is the auto-schema / auto-decomposition
        entry point: hand it a cube, get back which axes are boring scaffolding and
        which carry the payload. `coords={axis: values}` supplies real coordinates
        (e.g. timestamps) so irregular sampling can raise an axis's information;
        `categorical=[axes]` marks unordered-label axes. Kept negative: a HEURISTIC
        recommender with a conservative threshold (prefers INDEX) -- override BIND in
        the conjunctive-coding case, which it flags via the coupling number."""
        from holographic.sampling_and_signal.holographic_axisrole import analyze_axes
        return analyze_axes(data, coords=coords, categorical=categorical)

    def comparability_cost(self, data, axis, dim=256, seed=0):
        """MEASURE the price of binding a boring axis into content (holographic_axisrole):
        compares adjacent-slice similarity when the axis is INDEXED (raw slices) vs
        BOUND (each slice rotated by a distinct per-slice key, as folding the axis in
        would do). Returns {indexed_sim, bound_sim, collapse}: on a boring carrier
        the indexed similarity is high and the bound similarity collapses toward 0
        (private-subspace rotation), so `collapse` is the concrete similarity
        destroyed by the wrong role choice -- the thesis in one number, on your own
        data, against the strongest honest baseline (raw indexed similarity)."""
        from holographic.sampling_and_signal.holographic_axisrole import comparability_cost
        return comparability_cost(data, axis, dim=dim, seed=seed)

    def rectify_carrier(self, coords, content, n_out=None):
        """REPAIR a nearly-boring carrier axis into a clean uniform index
        (holographic_axisrole): a non-monotone axis (delta occasionally negative)
        is lifted by CUMULATIVE ARC LENGTH -- the monotone/covering-lift move from
        the sign-as-rotation work, absorbing small reversals into one-way progress;
        an irregular axis is then RESAMPLED onto a uniform grid by interpolation.
        Returns the uniform coords, resampled content, and the measured
        marginal-info before/after (after is 0.0 -- ideal carrier -- by
        construction, and returning both makes the repair auditable). Kept
        negatives: interpolation invents between samples (index repair, not
        payload information); a LARGELY reversing axis makes content a path, not
        a function of the axis -- monotone_fraction is returned so that regime is
        visible (below ~0.9: inspect by hand)."""
        from holographic.sampling_and_signal.holographic_axisrole import rectify_carrier
        return rectify_carrier(coords, content, n_out=n_out)

    def winding_map(self, coords, content, grid_size=100, agree_tol=0.15):
        """When a carrier LARGELY reverses (rectify_carrier's monotone_fraction well
        below ~0.9), is content a FUNCTION of the axis or a PATH over it?
        (holographic_winding). Splits the trajectory into monotone LAPS at
        reversals, interpolates each onto a shared grid over the revisited range,
        and measures lap agreement. Verdicts: 'function' (all laps agree -> merged,
        noise-averaged profile: multi-pass scanning is free denoise, ~sqrt(#laps));
        'hysteresis' (laps agree within a direction, differ across -> per-direction
        branches, the covering lift x -> (x, direction); merging REFUSED because
        the average is a curve no pass traced); 'path' (laps differ even within a
        direction: drift/aging -> per-lap curves, no merge). The disagreement
        numbers travel with every verdict. Completes the axis-role arc's declared
        boundary: the honest structure for a genuinely revisiting carrier."""
        from holographic.sampling_and_signal.holographic_winding import winding_map
        return winding_map(coords, content, grid_size=grid_size, agree_tol=agree_tol)

    def explore_series(self, data, coords=None, max_terms=6, auto_demux=False, cross_channel=False, handle_reversals=False):
        """AUTO-EXPLORE an unlabeled multi-axis series (holographic_scaffold): try
        every axis as the candidate scaffold (boring AND organising: score =
        continuity * (1 - marginal_info), full table returned); rectify the
        winner's coordinates when they wobble (rectify_carrier); decompose each
        payload channel along the carrier into its generating law
        (decompose_signal, MDL-gated); recompose and account variance -- each
        channel returns its explained fraction AND its residual, the honest
        hand-off to the next level of analysis. Verdict 'structured' / 'weakly
        structured' / 'no structure found' decided by measured explained
        fractions; noise is never dressed as law. Two default-off stages (old
        calls byte-identical): auto_demux=True runs demux_series first on a raw
        1-D stream (find the interleave stride, split, group into objects,
        explore each independently -- the Contact protocol with zero hints);
        cross_channel=True runs cross_channel_links on the RESIDUALS (delayed-
        copy structure per-channel decomposition cannot see; found links upgrade
        a bare 'no structure found' to 'weakly structured'); handle_reversals=
        True routes a largely-reversing carrier through winding_map (measured:
        multi-pass merge cuts profile RMS 2.4x; hysteresis/path verdicts are
        reported, never merged). Pure orchestration
        of shipped faculties -- hand the engine a raw stream, get back the
        sources, the schema, the laws, and the leftovers."""
        from holographic.sampling_and_signal.holographic_scaffold import explore_series
        return explore_series(data, coords=coords, mind=self, max_terms=max_terms,
                              auto_demux=auto_demux, cross_channel=cross_channel,
                              handle_reversals=handle_reversals)

    def decompose_piecewise(self, y, min_seg=16, penalty=3.0, max_terms=6):
        """Decompose a PIECEWISE signal (holographic_scaffold): segment at the
        statistics shifts first (segment_stream, the packet_demux instrument),
        then fit a law PER SEGMENT with decompose_signal. A regime-built signal
        fits a global formula badly (no 'switch at t' atom in the dictionary);
        per-regime fitting puts each law in its own house. MEASURED vs the
        global-fit baseline on a 3-regime signal: residual RMS 0.5001 -> 0.0013,
        MDL bits 2723 -> 588 (4.6x better compression). The result CARRIES its
        baseline (global residual + bits), so a signal where segmentation does
        not pay is visible as such. Kept scope: regime changes need a statistics
        shift; repeated regimes are paid for twice (no cross-segment parameter
        sharing); hard cuts at boundaries."""
        from holographic.sampling_and_signal.holographic_scaffold import decompose_piecewise
        return decompose_piecewise(y, mind=self, min_seg=min_seg, penalty=penalty,
                                   max_terms=max_terms)

    def packet_demux(self, x, min_seg=16, penalty=3.0, noise_k=3.0, continuation=False):
        """Demultiplex a PACKETIZED stream (holographic_demux): variable-length
        bursts from different sources -- the case the cyclic interleave scan
        (demux_series) declares out of scope. Stage 1: change-point segmentation
        by binary segmentation with a BIC-style penalty on a PIECEWISE-LINEAR
        cost (a homogeneous stream honestly returns no boundaries; drifting
        ramps do not shatter). Stage 2: NOISE-CALIBRATED assignment -- split-half
        signatures estimate each pair's own noise, features weighted by 1/noise
        with a shrinkage floor, segments merge within noise_k (3) times the
        pair's own self-distance: no magic threshold. Optional Stage 3
        (continuation=True, default OFF, old results byte-identical): reunite
        sources that DRIFT ACROSS bursts -- extrapolate each source's linear
        tail across the gap and merge when the later burst resumes at the
        predicted level AND slope AND with matching residual dynamics; every
        merge carries {predicted, observed, tolerance}. Returns boundaries,
        assignment, per-source REASSEMBLED streams ready for explore_series.
        Kept scope: a boundary needs a statistics shift; oscillating sources
        over-segment (model boundary); assignment is conservative (under-merging
        fabricates nothing); an impostor starting exactly on another source's
        extrapolation at its slope is indistinguishable by construction. The
        continuous costume of holographic_segment's discrete branching-entropy
        move."""
        from holographic.sampling_and_signal.holographic_demux import packet_demux
        return packet_demux(x, min_seg=min_seg, penalty=penalty, noise_k=noise_k,
                            continuation=continuation)

    def detect_regimes(self, x, min_seg=16, penalty=3.0):
        """WHERE does a recorded series change behaviour? Located change-point detection over a whole batch
        (holographic_demux.segment_stream): binary segmentation with a BIC-style penalty returns the exact
        boundary indices where the statistics shift, plus each segment's (start, stop, mean, std, length). A
        homogeneous stream honestly returns NO boundaries; a drifting ramp does not shatter.

        Complements regime_detector, it does NOT duplicate it: regime_detector is a CAUSAL ONLINE detector (sees
        one sample at a time, commits to a new layer when a fast/slow divergence persists -- for a live stream);
        detect_regimes is the OFFLINE batch twin (given the whole recording, locate every boundary at once). Use
        this to find where a query stream's statistics shifted so a cache margin can be re-fit per regime, to
        split a forecast at its regime boundaries instead of fitting one global model, or to segment any recorded
        engine signal (a trajectory, an error curve, an access trace) into homogeneous spans.

        Returns {boundaries: [int], n_segments: int, segments: [{start, stop, mean, std, length}]}. See
        holographic_demux.segment_stream."""
        import numpy as np
        from holographic.sampling_and_signal.holographic_demux import segment_stream
        x = np.asarray(x, float).ravel()
        seg = segment_stream(x, min_seg=min_seg, penalty=penalty)
        # enrich the raw (start, stop) spans with per-segment statistics -- the WHAT-changed alongside the WHERE,
        # so a caller (re-fit this cache regime, split this forecast) has the numbers without a second pass.
        segments = [{"start": int(a), "stop": int(b), "length": int(b - a),
                     "mean": float(x[a:b].mean()) if b > a else 0.0,
                     "std": float(x[a:b].std()) if b > a else 0.0}
                    for (a, b) in seg["segments"]]
        return {"boundaries": [int(bd) for bd in seg["boundaries"]],
                "n_segments": int(seg["n_segments"]), "segments": segments}

    def cross_channel_links(self, series, max_lag=None, threshold=0.6):
        """Find DELAYED-COPY / shared-component links between channels
        (holographic_demux): for every ordered pair, scan lags of the normalized
        cross-correlation; a strong peak at lag L with gain g means channel j ~
        g * channel i delayed by L -- structure invisible to per-channel
        decomposition (a delayed copy of noise decomposes to nothing on both
        channels, yet the pair is lawful together). The residual pass
        explore_series's leftovers were returned for. Direction falls out of
        which ordering carries the peak. Kept negatives: linear, pairwise,
        single-lag (time-varying delays, nonlinear couplings, three-way sources
        out of scope); a lag statement, not a mechanism claim."""
        from holographic.sampling_and_signal.holographic_demux import cross_channel_links
        return cross_channel_links(series, max_lag=max_lag, threshold=threshold)

    def demux_series(self, x, max_k=12, group_threshold=0.6):
        """ONE stream, MANY sources (holographic_demux): separate the channels,
        group the objects. A 1-D stream is tested for round-robin INTERLEAVING
        (the Contact move: sample i belongs to channel i mod K) -- the stride K is
        FOUND by delta-continuity (at the true K every strided sub-stream is a
        smooth single source; deinterleaving is a permutation, so recovery is
        bit-exact), with the smallest-K Occam rule over the m*K harmonic ladder
        and an honest K=1 when nothing separates. The channels (or an already
        multi-channel series) are then GROUPED into objects by |correlation| of
        their trajectories (absolute: a mirrored axis of one rigid motion still
        belongs to its mesh) -- a multi-object animated-mesh delta stream resolves
        into its meshes. Each returned object is ready for explore_series: decode
        each channel separately. Score table and correlation matrix travel as
        evidence. Kept scope: cyclic interleaving only (packetized muxing won't
        score a clean K); linear |corr| grouping (a time-varying axis mix can
        split an object -- the matrix is returned so it's visible)."""
        from holographic.sampling_and_signal.holographic_demux import demux_series
        return demux_series(x, max_k=max_k, group_threshold=group_threshold)

    def analytic_signal(self, x):
        """Decompose a signed series into its ROTATION (holographic_analytic): the
        analytic signal z = x + i*Hilbert(x), returned as amplitude (the instantaneous
        envelope / circle radius), unwrapped phase (how far it has rotated), and
        instantaneous frequency (how fast the sign turns over). A*cos(phase)
        reconstructs x exactly -- a lossless re-coordinatisation, not a model. NumPy-
        only Hilbert transform (no scipy). Use to get sign-aware, comparable
        coordinates for a signed signal, or to feed a phasor memory. Kept negative:
        edge effects near the ends (global FFT); meaningful mainly for narrowband /
        monocomponent signals."""
        from holographic.sampling_and_signal.holographic_analytic import analytic_signal
        return analytic_signal(x)

    def monotone_cost(self, x, direction=1):
        """MEASURE the price of clockwise-only (one-way) rotation on a REAL signed
        series (holographic_analytic). Reconstructs the signal with the full
        (reversible) phase vs a phase clamped to advance one way, and reports the
        excess error, the reversal fraction, and where the error concentrates. Kept
        finding (sharp): a real scalar signal is ALREADY a one-way rotation (its
        spectrum is symmetric, so instantaneous frequency is non-negative), so this
        honestly reads ~0 reversal fraction and small excess -- a single real channel
        cannot even carry a reversal. The real group-vs-monoid price lives on the
        complex/I-Q path: see phasor_monotone_cost."""
        from holographic.sampling_and_signal.holographic_analytic import monotone_cost
        return monotone_cost(x, direction=direction)

    def phasor_monotone_cost(self, z, direction=1):
        """MEASURE the group-vs-monoid price where it actually lives: a TRUE complex
        rotation (holographic_analytic). A complex/I-Q series carries a genuine
        rotation DIRECTION in its two channels and can truly reverse; clamping it
        clockwise-only then loses the reversal, at a large well-defined cost. This is
        the quadrature encoder with both channels present -- drop to one direction and
        you pay. Returns reversal_fraction, monotone_rmse, excess, max_local_error.
        The complement to monotone_cost (which reads ~0 on real signals, by the
        symmetric-spectrum theorem)."""
        from holographic.sampling_and_signal.holographic_analytic import phasor_monotone_cost
        return phasor_monotone_cost(z, direction=direction)

    def identify_dynamics(self, x=None, dt=None, force=None, positions=None, G=None, interaction=None):
        """Identify MASS / MOMENTUM / dynamics from measurements (holographic_sysid),
        via whichever honest door the supplied channels open: a FORCE channel (fit
        m*a + c*v + k*x = F -> mass, damping, stiffness, momentum, in the force's
        units); an INTERACTION (momentum conservation -> the mass RATIO, Mach's
        operational definition); or a KNOWN FORCE LAW with its constant (positions of
        a bound orbit + G -> the central mass by Kepler's third law -- how astronomy
        weighs stars with no force sensor). A trajectory ALONE raises GaugeError with
        the theorem: scaling mass and force together leaves the path bit-identical
        (F=ma exposes only F/m), so mass is unidentifiable -- kinematics (velocity,
        acceleration) is offered instead. Refuse-rather-than-guess, applied to
        physics. General across fields: lab carts (force), collider events
        (interaction), orbits (force law)."""
        from holographic.sampling_and_signal.holographic_sysid import identify
        return identify(x=x, dt=dt, force=force, positions=positions, G=G, interaction=interaction)

    def central_mass_from_orbit(self, positions, dt, G=6.674e-11):
        """Weigh a central body from a bound orbit (holographic_sysid): Kepler's
        third law M = 4*pi^2*a^3/(G*T^2), with the semi-major axis from the radius
        extremes and the period from the unwrapped bearing (the monotone-rotation /
        winding picture on a genuine 2-channel signal). Works for 2-D or inclined
        3-D orbits (best-fit plane by SVD). Refuses (GaugeError) on less than one
        full observed orbit rather than extrapolating a period. Kept scope: central
        body dominant (test mass), single body, Keplerian."""
        from holographic.sampling_and_signal.holographic_sysid import central_mass_from_orbit
        return central_mass_from_orbit(positions, dt, G=G)

    def diagnose_scaling(self, eval_fn, knobs, factor=2.0):
        """Detect WHICH limit a workload is hitting (holographic_scalinglaw): scale
        each declared knob (dim, tiles, bits, resolution, samples -- anything) by
        `factor` in isolation, measure the error response, and rank the levers. The
        house dim-doubling rule generalised to every resource and made executable:
        a limit is diagnosed by which knob's doubling reduces the error; a WALL is
        when no knob does (verdict 'wall' -- scaling is the wrong tool here, walk
        the five levers for a different approach instead). eval_fn(**knobs) must be
        deterministic and return error (float) or {'error','cost'}. Returns the
        probe table (the evidence), ranked knobs, and the verdict. Kept negative:
        first-order and local -- a knob that only pays at 4x or only jointly with
        another reads unresponsive here (auto_scale's re-probing recovers the
        first case)."""
        from holographic.misc.holographic_scalinglaw import diagnose_scaling
        return diagnose_scaling(eval_fn, knobs, factor=factor)

    def auto_scale(self, eval_fn, knobs, target_error, max_rounds=8, factor=2.0):
        """Automatic scaling (holographic_scalinglaw): repeatedly diagnose from the
        CURRENT operating point and double the most responsive knob, until the
        target error is met, a WALL is diagnosed (no knob helps -- stop and say so
        rather than burn the budget), or max_rounds is spent. Every step in the
        returned trajectory carries the probe that justified it -- no scaling
        decision without its baseline. The capacity-adaptive pattern (octree /
        load-gated record) generalised to any workload with declared knobs."""
        from holographic.misc.holographic_scalinglaw import auto_scale
        return auto_scale(eval_fn, knobs, target_error, max_rounds=max_rounds, factor=factor)

    def diagnose_bake(self, grids, values, queries=None, dim=4096, margin=1.5, seed=0):
        """For an n-D texture bake of THIS field, should you raise the DIMENSION or the BANDWIDTH (margin)? --
        measured, not guessed (holographic_scalinglaw). Wires diagnose_scaling to bake_nd on a held-out query set,
        so the engine's most-repeated tuning rule ('double D: if error drops you are variance-limited, else raise
        the bandwidth') becomes one call. Returns diagnose_scaling's dict, whose `verdict` is 'scale:dim' (more
        dimension pays) or 'scale:margin' (widen/narrow the kernel; more dimension is wasted), each carrying the
        measured per-knob error drop as its own evidence. Use before committing to an expensive high-dim bake.
        See holographic_scalinglaw.diagnose_bake."""
        from holographic.misc.holographic_scalinglaw import diagnose_bake
        return diagnose_bake(grids, values, queries=queries, dim=dim, margin=margin, seed=seed)


    def fractal_confidence(self, x, tol=0.15):
        """A singularity CROSS-CHECK for a 1-D signal's fractal dimension (holographic_bandwidth): two independent
        slope estimators -- the power-spectrum slope D=(5-gamma)/2 and an increment-variance estimator -- and
        whether they AGREE. The shipped fractal_dimension is a single estimator and silently returns a wrong number
        for a step or a pure tone; this flags those (agree=False). Returns (d_spectral, d_increment, agree). Trust a
        fractal dimension only when agree is True. (R/S Hurst is deliberately NOT a co-validator here -- it weights
        coarse scales differently and disagrees even on clean fBm.)"""
        from holographic.misc.holographic_bandwidth import fractal_confidence
        return fractal_confidence(x, tol=tol)

    def density_estimate(self, samples, lo, hi, query, dim=1024, seed=None, method="lcv", bandwidth=None):
        """Kernel DENSITY ESTIMATE via the encoder (holographic_kde): bundle the encoded samples, then density(x) ~
        bundle . encode(x) = (1/n) sum K(x - s_i), with the RBF kernel bandwidth AUTO-SELECTED (the band-limit
        matched to the data) and the output normalized to integrate to ~1 over [lo,hi]. method='lcv' (leave-one-out
        likelihood, robust -- beats a fixed default ~5-7x by matching the kernel to the data) or 'silverman' (cheap,
        over-smooths multimodal). Returns (density_at_query, chosen_bandwidth). The disciplined form of the
        band-limited-encoding faculty: the encoder's documented RBF-as-KDE use, where the bandwidth IS the
        band-limit. Kept negatives: the sinc kernel's bandwidth is NOT tunable (only RBF); bandwidth selection fixes
        smoothing, not capacity (a too-small dim cannot be rescued)."""
        from holographic.sampling_and_signal.holographic_kde import density_estimate
        return density_estimate(samples, lo, hi, query, dim=dim, seed=self.seed if seed is None else seed,
                                method=method, bandwidth=bandwidth)

    def kde_bandwidth(self, samples, lo, hi, method="lcv"):
        """The RBF kernel bandwidth for a density estimate over [lo,hi] (holographic_kde): 'lcv' (leave-one-out
        likelihood, robust, matches the data's structure) or 'silverman' (rule of thumb, over-smooths multimodal).
        The band-limit auto-matched to the data. Returns a float."""
        from holographic.sampling_and_signal.holographic_kde import kde_bandwidth
        return kde_bandwidth(samples, lo, hi, method=method)

    def spectral_flatness(self, v):
        """SPECTRAL FLATNESS of a vector (holographic_flatness): the Wiener entropy of its power spectrum (geometric
        mean / arithmetic mean), in (0,1]. 1.0 = a perfectly flat spectrum = a UNITARY, distortion-free binding key;
        lower = peakier = more lossy as a key. The diagnostic for "is this safe to bind/unbind repeatedly?" --
        unbind(bind(x,k),k) returns x convolved with |K|^2, which is x only when |K|=1 everywhere (flatness 1).
        Returns a float."""
        from holographic.misc.holographic_flatness import spectral_flatness
        return spectral_flatness(v)

    def binding_stability(self, v, tol=0.05):
        """BINDING-STABILITY regime diagnostic for a key (holographic_flatness): {'flatness', 'distortion',
        'stable'}, where distortion is the measured single-round bind/unbind error and 'stable' iff it is within tol
        (effectively unitary). Spectral flatness PREDICTS distortion monotonically. The band-limit-preservation
        regime test grounded in the real bind: the engine already mints the stable (unitary) regime via
        unitary_vector; this measures where any vector sits on it. Kept finding: the Trefethen transient-growth
        concern does not materialize -- linear ops preserve a white spectrum and the cleanup contracts monotonically;
        the real stability axis is this linear flatness."""
        from holographic.misc.holographic_flatness import binding_stability
        return binding_stability(v, tol=tol)

    def verify_image_structure(self, image, real_patches=None, patch=32):
        """Does an image carry the spatial-autocorrelation signature of real data
        (vs noise / corruption)? The text structure verifier, carried to images
        (holographic_signal_structure). If real_patches is None, calibrates on
        patches of the image itself. Returns {'score', 'structured', 'threshold'}."""
        from holographic.misc.holographic_signal_structure import SignalStructureVerifier
        img = np.asarray(image, float)
        if img.ndim == 3:
            img = img.mean(axis=2)
        if real_patches is None:
            h, w = img.shape
            real_patches = [img[i:i + patch, j:j + patch]
                            for i in range(0, max(1, h - patch), patch)
                            for j in range(0, max(1, w - patch), patch)] or [img]
        v = SignalStructureVerifier("image").calibrate(real_patches)
        return {"score": v.structure_score(img), "structured": bool(v.is_structured(img)),
                "threshold": float(v.threshold)}

    def volatility_structure(self, returns):
        """Does a return series carry the volatility-clustering signature of real
        markets (|returns| autocorrelated)? Returns the clustering z-score vs a
        shuffled control: >2 is meaningful structure, near 0 means none (or too
        little data). The cross-domain structure verifier for time series."""
        from holographic.misc.holographic_signal_structure import clustering_zscore, volatility_clustering
        r = np.asarray(returns, float).ravel()
        return {"clustering": float(volatility_clustering(r)),
                "zscore": float(clustering_zscore(r))}

    def resolution_profile(self, x, modality=None, among=None):
        """How much holographic RESOLUTION does classifying this input need? For
        each truncation dimension, which prototype wins -- and at what dimension
        does the winner stabilise? A low stabilisation dimension means the answer
        is robust to heavy truncation (a 'fundamental' match); needing full width
        means it was a close call. The persistent-homology idea made practical:
        which structure survives compression. Returns
        {'profile': [(dim, label, score)], 'stable_from': dim, 'full_dim': D}."""
        from holographic.misc.holographic_resolution import resolution_profile as _rp
        v = self.perceive(x, modality)
        protos, labels = [], []
        for lab, _, unit, _ in self.memory.live._p:
            if among is None or lab in among:
                protos.append(unit)
                labels.append(lab)
        if not protos:
            return {"profile": [], "stable_from": 0, "full_dim": self.dim}
        M = np.stack(protos)
        prof = _rp(v, M)
        named = [(k, labels[i], round(s, 3)) for k, i, s in prof]
        final = prof[-1][1]
        stable = prof[-1][0]
        for k, i, _ in prof:
            if i == final:
                stable = k
                break
        return {"profile": named, "stable_from": stable, "full_dim": self.dim}

    def read_role(self, label, role):
        """Decode one role's filler from a LEARNED class -- unbind the role from
        the class prototype and clean up against the experience-registered
        fillers. Works whether the class holds one record or the superposition
        of many noisy ones (measured: see explain)."""
        from holographic.agents_and_reasoning.holographic_ai import bind, involution
        est = bind(self._class_vec(label), involution(self.encoder._roles.get(str(role))))
        return self._clean_filler(est, role)

    def ask(self, start_filler, *path):
        """A CHAIN over the mind's own memory: ask('paris', ('capital',
        'currency'), ('currency', 'language')) -> the language of the country
        with the currency of the country whose capital is paris. Each hop is
        find() then read() -- geometry snapped to a symbol before the next hop,
        which is what keeps chains exact instead of compounding HRR noise."""
        filler = start_filler
        for match_role, read_role in path:
            label, _ = self.find(match_role, filler)
            if label is None:
                return None
            filler, _ = self.read_role(label, read_role)
        return filler

    def blend(self, base_label, donor_label, donor_roles):
        """PROJECTION TO CREATE NEW THINGS, over the mind's OWN learned classes.
        Synthesize a novel concept: the frame of `base_label`, with `donor_label`'s
        values projected onto `donor_roles`. The mind decodes each role from its
        class prototypes (so this works over concepts learned from many noisy
        observations, not just hand-built records) and rebuilds a coherent new
        record that names a thing it never saw -- 'this class, but with that
        class's distinctive traits'. Returns {role: value} for the synthesized
        concept. (Analogy as CREATION: synthesizing a specified new thing is
        well-posed and exact where RETRIEVING an existing analogue from a clean
        role-filler memory is not -- every learned class is an exact key, so there
        is no graded nearness for a retrieval-analogy to climb.)"""
        donor_roles = set(donor_roles)
        roles = sorted(self._fillers)
        spec = {}
        for r in roles:
            src = donor_label if r in donor_roles else base_label
            val, _ = self.read_role(src, r)
            if val is not None:
                spec[r] = val
        return spec

    def ask_traced(self, start_filler, *path, min_throughput=0.0):
        """ask() instrumented like a PATH TRACER: a relation chain is a ray
        bouncing through the holographic space, each hop a bounce whose cleanup
        confidence is its reflectance, and throughput is the accumulated product.
        Throughput is a calibrated confidence in the chained answer (measured:
        keeping only the most-confident chains sharply raises accuracy on the
        answered subset), and a chain whose throughput decays below
        `min_throughput` ABSTAINS (returns answer None) rather than emitting
        noise -- the energy-based termination of a ray that has lost too much to
        contribute. Returns (answer_or_None, throughput, hop_confidences)."""
        filler = start_filler
        throughput = 1.0
        confidences = []
        for match_role, read_role in path:
            label, fconf = self.find(match_role, filler)
            if label is None:
                return None, 0.0, confidences
            filler, rconf = self.read_role(label, read_role)
            hop = max(0.0, float(fconf)) * max(0.0, float(rconf))
            throughput *= hop
            confidences.append(round(hop, 3))
            if throughput < min_throughput:
                return None, throughput, confidences
        return filler, throughput, confidences

    def explain(self, x1, x2):
        """WHY are two things similar -- not just a cosine, but the per-role
        verdict. Takes either two record DICTS (encoded fresh, candidates drawn
        from the inputs) or two LEARNED LABELS (decoded from the mind's own
        class prototypes, candidates from the experience-registered fillers --
        so the mind explains concepts it learned, including classes built from
        many noisy observations).

        Returns [(role, value_1, value_2, shared, confidence), ...]. Built on
        the measured relations operations (per-role explanation 4/4, naming
        100%, symbol-routed mapping 360/360, chains exact through three hops);
        every readout cleans up to a SYMBOL, because meaning survives
        composition only when it touches symbols between steps."""
        from holographic.agents_and_reasoning.holographic_ai import bind, involution, cosine
        if isinstance(x1, dict) and isinstance(x2, dict):
            rec1 = self.encoder.encode(x1, "record")
            rec2 = self.encoder.encode(x2, "record")
            values = sorted({str(v) for v in list(x1.values()) + list(x2.values())})
            val_vecs = {v: self.encoder.encode(v) for v in values}

            def clean(vec):
                best, score = None, -2.0
                for v, vv in val_vecs.items():
                    s = cosine(vec, vv)
                    if s > score:
                        best, score = v, s
                return best, float(score)

            out = []
            for role in sorted(set(x1) & set(x2), key=str):
                inv = involution(self.encoder._roles.get(str(role)))
                f1, c1 = clean(bind(rec1, inv))
                f2, c2 = clean(bind(rec2, inv))
                out.append((str(role), f1, f2, f1 == f2, min(c1, c2)))
            return out
        # learned labels: decode from the class prototypes the mind built itself
        v1, v2 = self._class_vec(x1), self._class_vec(x2)
        out = []
        for role in sorted(self._fillers):
            inv = involution(self.encoder._roles.get(role))
            f1, c1 = self._clean_filler(bind(v1, inv), role)
            f2, c2 = self._clean_filler(bind(v2, inv), role)
            out.append((role, f1, f2, f1 == f2, min(c1, c2)))
        return out

    def finding_registry(self):
        """The findings registry (backlog D3): a research log as a holographic KNOWLEDGE STRUCTURE. Record
        structured claims -- a SUBJECT affects an OBJECT with a +1/-1 POLARITY (helps / hurts), optionally
        under a CONDITION (a regime) -- then recall them by similarity and detect the log's OWN
        contradictions. It extends the relations layer (the same role-bound records KnowledgeStore's
        explain/analogy run on) with the piece a research log needs: distinguishing a FLAT contradiction
        (two opposite-polarity findings about the same claim under the same/absent condition -- one must be
        wrong) from a CONDITIONED tension (the same under DIFFERENT conditions -- reconcilable, the outcome is
        conditioned on the differing dimension). Retrieval is holographic (cosine over the bound claim); the
        verdict is exact (polarity sign, condition equality). Lazily created and cached on the mind's own
        dimension/seed; call .add / .query / .tensions on the returned FindingRegistry.

        SCOPE (kept negative): findings are STRUCTURED claims, not free prose -- turning narrative into
        structured claims is an NLP step this engine does not do (no embeddings, no parser)."""
        reg = getattr(self, "_finding_registry", None)
        if reg is None:
            from holographic.agents_and_reasoning.holographic_knowledge import FindingRegistry
            reg = FindingRegistry(dim=self.dim, seed=self.seed)
            self._finding_registry = reg
        return reg

    def explain_splits(self, label, contrast_floor=0.25):
        """INCEPTION: the mind explains its own memory organization. When the
        self-organizing memory has split `label` into sub-prototypes (because
        held-out accuracy said the class is genuinely multi-modal), each
        sub-prototype is a superposition of one MODE's records -- so decoding
        the registered roles from each names WHAT the split separated: 'this
        class divided because one mode is colour=red and the other colour=blue'.
        The relations machinery (built on the substrate) explaining the
        organizer (built on the same substrate).

        A role counts as SEPARATING only by CONTRAST -- each mode's winning
        value must be genuinely absent from the other mode, not merely less
        common. Measured on the XOR world: truly separating roles score ~0.5
        contrast, incidental skews (a noise role one 2-means half happened to
        lean toward) score <= 0.1, so the floor sits mid-gap. The statistic's
        first real outing caught the organizer red-handed: one label's split
        turned out to separate the NOISE role (accuracy-sufficient, since the
        other label's clean split already resolved the XOR) -- the explanation
        honestly reports what the split actually did, not what was assumed.

        Returns (decodes, separating): per-sub-prototype {role: (value, score)}
        and the roles whose contrast clears the floor. A single-prototype label
        returns an empty separating set (nothing was divided, which is itself
        the explanation)."""
        from holographic.agents_and_reasoning.holographic_ai import bind, involution, cosine
        subs = [unit for lab, _, unit, _ in self.memory.live._p if lab == label]
        if not subs:
            raise KeyError(f"unknown label: {label!r}")
        # full per-role value scores for every sub-prototype
        scores = []
        for u in subs:
            row = {}
            for role in sorted(self._fillers):
                est = bind(u, involution(self.encoder._roles.get(role)))
                row[role] = {v: cosine(est, self.encoder.encode(v))
                             for v in self._fillers[role]}
            scores.append(row)
        decodes = [{r: max(row[r].items(), key=lambda kv: kv[1]) for r in row}
                   for row in scores]
        separating = []
        if len(subs) > 1:
            for role in sorted(self._fillers):
                winners = [d[role][0] for d in decodes]
                if len(set(winners)) < 2:
                    continue
                # contrast: my winner's score here, minus every OTHER mode's
                # winner scored here -- averaged over modes (mutual absence)
                cs = []
                for i, row in enumerate(scores):
                    own = row[role][decodes[i][role][0]]
                    others = [row[role].get(decodes[j][role][0], 0.0)
                              for j in range(len(subs)) if j != i]
                    cs.append(own - max(others))
                if float(np.mean(cs)) >= contrast_floor:
                    separating.append(role)
        return decodes, separating

    def explain_organization(self):
        """The whole memory's self-explanation: for every label the organizer
        split, the nameable reason. {label: differing_roles}."""
        out = {}
        for label in sorted(self.memory.live.labels()):
            subs = sum(1 for lab, *_ in self.memory.live._p if lab == label)
            if subs > 1:
                _, differing = self.explain_splits(label)
                out[label] = differing
        return out

    def classify_robust(self, x, modality=None, route=True, n_rays=5, seed=0):
        """MULTI-RAY classification: one query is one noisy ray; fire several
        independent encodings and combine them, the way path tracing fires many
        rays per pixel and averages (no single ray is reliable, but the ensemble
        converges). For text the independent views are word-resampled subsets of
        the query -- each a different SHADOW of the same input -- and the crucial
        step is that each ray's per-label scores are Z-SCORED before summing, so a
        ray that is confident-but-wrong (an outlier view) cannot dominate, exactly
        the failure that sinks a naive vote. Measured: on a task where the views'
        individual accuracy ranges wildly (100%/100%/50%/17% across feature
        lenses), the z-scored ensemble reaches the BEST single view's accuracy
        BLIND -- without being told which view to trust.

        Falls back to plain classify() for non-text inputs (where word-resampling
        does not apply) and for single-token queries (no subset to resample).
        Returns (label, agreement) where agreement is the fraction of rays that
        voted for the winner -- a multi-ray confidence."""
        import numpy as np
        if modality is None:
            modality = self.encoder.infer(x)
            if modality == "text":
                modality = self._resolve_text_like(x)
        if modality not in ("text", "code") or not isinstance(x, str):
            lab, _ = self.classify(x, modality, route)
            return lab, 1.0
        words = x.split()
        if len(words) < 3:
            lab, _ = self.classify(x, modality, route)
            return lab, 1.0
        among = None
        if route:
            among = {lab for lab, m in self._label_modality.items()
                     if m == modality} or None
        rng = np.random.default_rng(seed)
        agg, raw_votes = {}, []
        views = [x] + [" ".join([w for w in words if rng.random() > 0.25] or words)
                       for _ in range(n_rays - 1)]
        for view in views:
            v = self.perceive(view, modality)
            sc = self.memory.live.label_scores(v, among=among)
            if not sc:
                continue
            vals = np.array(list(sc.values()))
            mu, sd = vals.mean(), vals.std() + 1e-9
            for lab, s in sc.items():               # z-score this ray's evidence
                agg[lab] = agg.get(lab, 0.0) + (s - mu) / sd
            raw_votes.append(max(sc, key=sc.get))
        if not agg:
            lab, _ = self.classify(x, modality, route)
            return lab, 1.0
        winner = max(agg, key=agg.get)
        agreement = raw_votes.count(winner) / max(1, len(raw_votes))
        return winner, round(agreement, 3)

    def recall(self, x, modality=None, abstain=None):
        """Nearest stored individual. The index does an exact scan until the store is
        genuinely big, then switches to the recursive HoloForest (the crossover is
        measured -- see _Index.recall). A NEGATIVE worth recording here: wiring the
        learned adaptive navigator (holographic_navigator) into this path was tried
        and lost badly on the mind's own store -- 48% recall@1 at ~130 comparisons,
        where the forest at beam 2 gets 89% within ~512. The navigator's margin
        senses were tuned on UNIFORM random vectors; the unified store is clustered
        (many near-duplicates per class), which miscalibrates the arrive/keep-moving
        instinct. So recall keeps the dumb-but-honest index, and the navigator stays
        a study of adaptive access, not a default.

        With `abstain` set (a false-alarm level alpha), returns the recalled payload only if the match is
        calibrated-significant (p <= alpha against the store's own noise floor, recall_calibrated) and None
        otherwise -- an honest 'I have nothing like this'. Default None preserves the original behaviour."""
        if self._recall is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        if abstain is not None:
            payload, _sim, p = self.recall_calibrated(x, modality=modality)
            return payload if (p == p and p <= abstain) else None
        return self._recall.recall(self.perceive(x, modality))

    # -- honest recall: calibrated confidence + abstention, woven into recognition --------
    # The honesty layer (RecallNull / SPRT / bh_fdr) was a standalone measurement harness; here it
    # becomes part of how the mind RECOGNISES. A raw cosine means nothing on its own -- RecallNull asks
    # how high pure noise reaches against THIS mind's own prototypes, turning a recall into an honest
    # false-alarm probability (the radio-SETI / particle-physics 'prove it isn't an artifact of your own
    # pipeline' discipline, calibrated to this mind's geometry). That p-value is what lets the mind
    # ABSTAIN -- say 'I don't recognise this' -- instead of always returning a nearest label, and it
    # upgrades the organizer's fixed-floor novelty heuristic to a calibrated one.
    def _recognition_null(self, n_null=1200):
        """Maintain a RecallNull over the CURRENT class-prototype codebook -- the mind's own noise floor.
        Rebuilt only when the prototype set changes (keyed on the store's mutation counter _gen), so
        steady-state recognition pays nothing. Returns None when nothing is learned yet."""
        labels, mat = self.memory.live._stack()
        if getattr(mat, "shape", (0,))[0] == 0:
            return None
        gen = getattr(self.memory.live, "_gen", 0)
        cache = getattr(self, "_null_cache", None)
        if cache is None or cache[0] != gen or cache[1] != mat.shape[0]:
            from holographic.agents_and_reasoning.holographic_honesty import RecallNull
            self._null_cache = (gen, mat.shape[0], RecallNull().fit(mat, n_null=n_null, seed=self.seed))
        return self._null_cache[2]

    def _resolve_modality(self, x, modality):
        """The modality classify() uses: declared, or inferred (with text-like sub-format resolution)."""
        if modality is None:
            modality = self.encoder.infer(x)
            if modality == "text":
                modality = self._resolve_text_like(x)
        return modality

    def recognize(self, x, modality=None, route=True):
        """CORE calibrated recognition. Like classify, but returns (label, similarity, pvalue): the
        pvalue is the honest false-alarm probability -- the chance pure noise would match the mind's own
        prototypes this well (RecallNull, calibrated to THIS mind). p small -> trust the label; p large ->
        the input matches no learned class. This is the basis of honest abstention and of the calibrated
        batch / streaming recognisers below."""
        modality = self._resolve_modality(x, modality)
        among = None
        if route:
            among = {lab for lab, m in self._label_modality.items() if m == modality} or None
        scores = self.memory.live.label_scores(self.perceive(x, modality), among=among)
        if not scores:
            return (None, 0.0, 1.0)
        label = max(scores, key=scores.get); sim = float(scores[label])
        null = self._recognition_null()
        p = float(null.pvalue(sim)) if null is not None else float("nan")
        return (label, sim, p)

    def _match_scores(self, window=600):
        """Genuine-match score density: recent stored examples' cosine to their OWN label's prototype
        (the quantity the organizer's coherence() reads). Feeds SPRT's match density from the mind's own
        experience -- no external calibration data."""
        recent = self.memory.buffer[-window:]
        out = [float(max((p[2] @ v for p in self.memory.live._p if p[0] == label), default=0.0))
               for v, label in recent]
        out = [s for s in out if s > 0.0]
        return np.asarray(out) if out else np.asarray([0.6])

    def stream_recognize(self, cues, modality=None, alpha=0.05, beta=0.05, route=True, cap=None):
        """Sequential recognition over a STREAM of cues bearing on the SAME thing (repeated noisy
        sightings of a landmark, a drifting pattern). Accumulates each cue's best-match score with Wald's
        SPRT and decides MATCH / REJECT the instant the evidence crosses a boundary -- the minimum expected
        number of samples for the (alpha, beta) error pair (decide as fast as the evidence allows). Returns
        (decision, recalled_label, n_samples_used). Null density = the mind's noise floor; match density =
        its own examples' self-similarity."""
        null = self._recognition_null()
        if null is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        from holographic.agents_and_reasoning.holographic_honesty import SPRTRecall
        sprt = SPRTRecall(null.null, self._match_scores(), alpha=alpha, beta=beta)
        votes, scores = {}, []
        for c in cues:
            lab, sim, _p = self.recognize(c, modality=modality, route=route)
            scores.append(sim)
            if lab is not None:
                votes[lab] = votes.get(lab, 0) + 1
        decision, n = sprt.decide(scores, cap=cap)
        return (decision, (max(votes, key=votes.get) if votes else None), n)

    def recognize_batch(self, queries, modality=None, alpha=0.1, route=True):
        """Recognise a BATCH honestly: classify each query, then control the FALSE-DISCOVERY RATE across
        the batch with Benjamini-Hochberg/Yekutieli (bh_fdr) over the per-query false-alarm p-values.
        Returns a list of {label, similarity, pvalue, significant}, where `significant` is True only for
        recognitions that survive FDR at `alpha` -- so scanning many queries cannot manufacture matches by
        luck (the look-elsewhere discipline applied to the mind's own recognition)."""
        from holographic.agents_and_reasoning.holographic_honesty import bh_fdr
        res = [self.recognize(q, modality=modality, route=route) for q in queries]
        pvals = np.asarray([(p if p == p else 1.0) for (_l, _s, p) in res])
        reject, _k = bh_fdr(pvals, alpha=alpha, dependent=True)
        return [{"label": l, "similarity": s, "pvalue": p, "significant": bool(r)}
                for (l, s, p), r in zip(res, reject)]

    def scan(self, channels, modality=None, alpha=0.05, beta=0.05, fdr=0.1, route=True, cap=None):
        """Scan MANY candidate channels with the two disciplines a large search needs AT ONCE -- Siemion's
        'flag anything that isn't noise' over an astronomical channel count, combining the streaming detector
        (B3) and the look-elsewhere control (FDR). Each channel is a STREAM of cues bearing on ONE hypothesis
        (a frequency bin over time, a sky position, a recurring market pattern). For every channel, Wald's
        SPRT decides MATCH/REJECT as fast as THAT channel's own evidence allows (the minimum expected samples
        for the (alpha, beta) error pair -- decide as fast as the evidence lets you); then
        Benjamini-Hochberg/Yekutieli FDR controls the trials factor ACROSS the channels (scan enough and some
        clear the per-channel bar by luck). A channel is a CONFIRMED detection (`detected`) only when the SPRT
        decided MATCH *and* its calibrated p-value survives FDR -- streaming detection and look-elsewhere in
        one pass. The per-channel p-value is calibrated for the EXACT statistic used: the channel's mean score
        against a null of equal-length noise streams, resampled from the per-cue noise floor (so a long
        channel and a short one are each judged against their own-length null, and the FDR is honest).
        Returns a per-channel list of {index, label, decision, n_samples, pvalue, fdr_significant, detected}.
        """
        null = self._recognition_null()
        if null is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        from holographic.agents_and_reasoning.holographic_honesty import SPRTRecall, RecallNull, bh_fdr
        match = self._match_scores()
        # routing the channels share (resolve once; recognize() recomputes the same per cue). Random noise
        # perceives to a ~random unit vector regardless of modality, so the floor below is modality-agnostic.
        rep_mod = modality
        if rep_mod is None and channels and len(channels[0]):
            rep_mod = self._resolve_modality(channels[0][0], None)
        among = ({lab for lab, mm in self._label_modality.items() if mm == rep_mod} or None) if route else None
        floor = self._scan_cue_null(rep_mod, route)       # PROCEDURE-MATCHED per-cue noise floor (see below)
        # The null for a channel's MEAN score is the mean of L i.i.d. draws from that per-cue floor. Resample
        # it from the calibrated floor (cheap) and cache by length L; seed per L so the draw is independent of
        # channel order (deterministic, Macklin's tie-break discipline).
        mean_null_cache = {}
        def mean_null(L):
            if L not in mean_null_cache:
                r = np.random.default_rng(self.seed + L)
                means = floor[r.integers(0, len(floor), size=(4000, L))].mean(axis=1)
                rn = RecallNull(); rn.null = np.sort(means); mean_null_cache[L] = rn
            return mean_null_cache[L]

        rows = []
        for i, cues in enumerate(channels):
            scored = [self.recognize(c, modality=modality, route=route) for c in cues]
            sims = [float(s) for (_l, s, _p) in scored]
            decision, n = SPRTRecall(floor, match, alpha=alpha, beta=beta).decide(sims, cap=cap)
            votes = {}
            for (lab, _s, _p) in scored:
                if lab is not None:
                    votes[lab] = votes.get(lab, 0) + 1
            label = max(votes, key=votes.get) if votes else None
            pval = float(mean_null(len(sims)).pvalue(float(np.mean(sims)))) if sims else 1.0
            rows.append({"index": i, "label": label, "decision": decision,
                         "n_samples": int(n), "pvalue": pval})
        reject, _k = bh_fdr([r["pvalue"] for r in rows], alpha=fdr, dependent=True)
        for r, sig in zip(rows, reject):
            r["fdr_significant"] = bool(sig)
            r["detected"] = bool(r["decision"] == "MATCH" and sig)
        return rows

    def detect_drifting(self, waterfall, drifts=None, alpha=0.01, off=None):
        """Find a DRIFTING narrowband signal in a spectrogram -- the SETI detection problem (Tarter,
        Siemion seats) cast in the engine's OWN primitives. A Doppler frequency drift is a cyclic
        SHIFT of the spectrum over time, and the engine already shifts: `permute` is exactly the
        rigid-shift transform holographic_video.py uses for motion compensation (and a shift is also
        a binding, bind(x, delta_k) == permute(x, k)). So "de-Doppler integration" -- the matched
        filter that recovers a drifting signal a STATIONARY detector loses -- is permute-ing each
        frame back by the drift before summing. The look-elsewhere control over the (drift x channel)
        search grid is `bh_fdr` with the DEPENDENT correction (the drift cells overlap; the honest,
        conservative choice). Supply `off` (an OFF-target spectrogram in the ON-OFF cadence radio
        astronomers use) to reject stationary RFI: a real signal is ON-only, terrestrial interference
        persists across the cadence.

        `waterfall` is (T frames x F bins). Returns detections [{drift, channel, snr, pvalue}, ...]
        sorted by SNR. Deterministic. MEASURED at the field's S/N>=10 regime: ~96% recall at 0%
        false-positive; the cadence rejects a strong stationary RFI ~100% while keeping the drifting
        signal ~94%. KEPT NEGATIVE: below ~10 sigma integrated the dependent-FDR correction over the
        many cells is conservative (a lone weak signal needs ~5 sigma) -- matching turboSETI's own
        S/N>=10 search threshold, which exists for exactly this reason.
        """
        from holographic.sampling_and_signal.holographic_dedoppler import detect_drifting as _dd
        off_arr = None if off is None else np.asarray(off, float)
        return _dd(np.asarray(waterfall, float), drifts=drifts, alpha=alpha, off=off_arr)

    def _scan_cue_null(self, modality, route, n=1500):
        """Procedure-matched per-cue noise floor for `scan`: the distribution of recognize()'s OWN sim on
        random unit vectors, taken through the SAME path a channel cue takes -- perceive() then the routed
        label_scores. This matters: perceive() is NOT the identity even for the 'vector' modality (it lifts a
        raw vector onto the encoder geometry, raising the max label score of pure noise from ~0.086 to ~0.117),
        and the recognition codebook's RecallNull scores prototype ROWS rather than the max label score
        recognize() returns -- calibrating the channel p-value to either of those wrong floors makes pure-noise
        channels look significant (a kept lesson). Calibrated for the encoded / vector-channel regime (the
        SETI spectrogram case); cached on the prototype generation, the modality, and route."""
        gen = getattr(self.memory.live, "_gen", 0)
        nproto = getattr(self.memory.live._stack()[1], "shape", (0,))[0]
        key = (gen, nproto, modality, bool(route))
        cache = getattr(self, "_scan_floor_cache", None)
        if cache is None or cache[0] != key:
            rng = np.random.default_rng(self.seed + 99991)
            sims = np.empty(n)
            for i in range(n):
                v = rng.standard_normal(self.dim); v /= np.linalg.norm(v) + 1e-12
                sims[i] = self.recognize(v, modality=modality, route=route)[1]
            self._scan_floor_cache = (key, np.sort(sims))
        return self._scan_floor_cache[1]

    def _recall_null(self, n_null=800):
        """The noise floor for 'have I stored anything actually like this?' -- and it is PROCEDURE-MATCHED:
        it is fit by running the SAME recall path (recall() -- the sublinear forest on a big store, the exact
        scan on a small one) on random unit queries and recording the score each reaches. Calibrated by
        construction (the null IS the score distribution noise produces under the real procedure), and it
        inherits recall()'s sublinearity, so it neither under-samples the store nor defeats the acceleration
        structure -- the two problems the earlier exact-scan + sampled-null version had. Cached on store size.
        Stored vectors and queries are unit length, so the index's dot is a cosine. Returns None if empty."""
        if self._recall is None or not getattr(self._recall, "vecs", None):
            return None
        n = len(self._recall.vecs)
        cache = getattr(self, "_recall_null_cache", None)
        if cache is None or cache[0] != n:
            rng = np.random.default_rng(self.seed)
            Q = rng.standard_normal((n_null, self.dim))
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
            scores = np.array([self._recall.recall(q)[1] for q in Q])   # the actual recall path on noise
            from holographic.agents_and_reasoning.holographic_honesty import RecallNull
            rn = RecallNull(); rn.null = np.sort(scores)                 # reuse its searchsorted pvalue
            self._recall_null_cache = (n, rn)
        return self._recall_null_cache[1]

    def recall_calibrated(self, x, modality=None):
        """CORE calibrated recall: the nearest STORED INDIVIDUAL plus an honest false-alarm probability --
        (payload, similarity, pvalue). p small -> the store really contains something like this; p large ->
        it does not (abstain). The symmetric partner of recognize() (which calibrates the class PROTOTYPE
        readout); recall(..., abstain=alpha) thresholds on it. The winner comes through recall() itself --
        the sublinear HoloForest on a big store, the exact scan on a small one -- so honest abstention does
        NOT cost the acceleration structure, and the null is matched to the same path (see _recall_null)."""
        if self._recall is None or not getattr(self._recall, "vecs", None):
            raise RuntimeError("nothing learned yet -- call learn() first")
        q = self.perceive(x, modality)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn                                  # unit query: stored vecs are unit, so the dot is a cosine
        payload, score = self._recall.recall(q)         # sublinear (forest) when big, exact when small
        null = self._recall_null()
        p = float(null.pvalue(float(score))) if null is not None else float("nan")
        return payload, float(score), p

    def combine_estimators(self, pairs, power=1.0):
        """Veach's balance/power heuristic (MIS): combine several estimators of the SAME quantity, weighting
        each by its per-query RELIABILITY (the 'pdf' analog). `pairs` is a list of (estimate, reliability);
        returns sum_i w_i * estimate_i with w_i = r_i**power / sum_j r_j**power (power=1 = the balance
        heuristic, 2 = the power heuristic). The reusable combiner behind mis_recover -- the point being that
        this BEATS a naive average, which instead carries each estimator's error into the other's weak regime
        (holographic_mis)."""
        from holographic.rendering.holographic_mis import combine_estimators
        return combine_estimators(pairs, power=power)

    def mis_recover(self, x, codebook, beta=10.0, power=1.0):
        """Recover a vector by combining HARD 1-NN and SOFT (dense-Hopfield) cleanups per-query via the balance
        heuristic -- the B1 kept-negative ('hard wins on discrete atoms, soft wins on continuous off-grid
        values') turned into one combiner that needs NO regime label. The cosine distribution's peakiness is
        the reliability: a sharp single winner trusts the exact atom; a close runner-up trusts the interpolating
        soft blend. `codebook` is the matrix of atoms to recover against. Measured to beat naive averaging
        always, and both singles in the crossover regime where neither estimator dominates (holographic_mis)."""
        from holographic.rendering.holographic_mis import mis_recover
        return mis_recover(np.asarray(x, float), np.asarray(codebook, float), beta=beta, power=power)

    def gradient_cache(self, anchors, values, jacobians):
        """Package sparse anchors with cached values AND local Jacobians for first-order (Ward irradiance-
        gradient) decode (CACHE-1). `anchors` (N,d), `values` (N,) or (N,M), `jacobians` (N,d) or (N,M,d). Read
        back with cache_interp -- the gradient lets each anchor cover more ground, ~halving the anchor count a
        smooth decode needs vs a nearest-neighbour grid argmax. (Use holographic_cache.gradient_cache_fd to
        build from a field function alone, with finite-difference Jacobians.)"""
        from holographic.caching_and_storage.holographic_cache import gradient_cache
        return gradient_cache(anchors, values, jacobians)

    def cache_interp(self, cache, q, validity_radius, global_weights=False):
        """Read a gradient cache at query `q` by first-order interpolation with a VALIDITY-RADIUS locality guard
        (CACHE-1): each anchor within the radius extrapolates its linear model v_i + J_i.(q-a_i), blended by
        1/distance. The guard is load-bearing -- global_weights=True removes it and a distant anchor dumps a bad
        long-range extrapolation into the query (measured ~2.7x worse); kept callable so the failure is visible."""
        from holographic.caching_and_storage.holographic_cache import interp_first_order
        return interp_first_order(cache, q, validity_radius, global_weights=global_weights)

    def optimize(self, loss, x0, grad=None, steps=200, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
                 tol=0.0, patience=10, min_steps=20, fd_eps=1e-5, stats=None):
        """GENERAL GRADIENT-DESCENT OPTIMIZER (holographic_optimize, GRAD-2) -- minimize any scalar `loss(x)` from
        `x0` by Adam (the exact bias-corrected update the splat fit uses, generalized). Supply `grad(x)` for the
        analytic gradient (fast); omit it for the finite-difference fallback (2*n loss evaluations per step). This
        promotes the 3DGS work's gradient machinery -- the splat fit's hand-derived-gradient Adam, plus the cache
        module's finite differences -- to a first-class engine capability: gradients on the fly, with no autodiff
        (the NumPy-only rule). `tol > 0` enables convergence-gated early stop; pass stats={} for steps and the loss
        trajectory. Underpins IHT recovery (GRAD-1) and any fit/alignment problem. Kept negative: no autodiff (FD is
        the general fallback and costs 2*n evals/step); gradient descent finds a LOCAL minimum on a non-convex loss."""
        from holographic.misc.holographic_optimize import optimize
        return optimize(loss, x0, grad=grad, steps=steps, lr=lr, b1=b1, b2=b2, eps=eps,
                        tol=tol, patience=patience, min_steps=min_steps, fd_eps=fd_eps, stats=stats)

    def fd_gradient(self, f, x, eps=1e-5):
        """The central finite-difference gradient of a scalar function f: R^n -> R at x (holographic_optimize,
        GRAD-2) -- 2*n evaluations, perturbing a copy one coordinate at a time. The general scalar-loss companion to
        the cache module's field-map finite differences; the 'gradient' for optimize() when no analytic one exists."""
        from holographic.misc.holographic_optimize import fd_gradient
        return fd_gradient(f, x, eps=eps)

    def adaptive_anchors(self, x, y, n, floor=0.05, power=0.5):
        """Place n cache/codebook anchor positions along `x` so they crowd where the field `y` bends (high
        curvature) and thin out where it is flat -- irradiance caching's adaptive record density instead of a
        uniform grid (CACHE-3). Density ~ |y''|^power equidistributes the piecewise-linear reconstruction error, so
        this matches uniform-placement quality at MATERIALLY fewer anchors (~7x on a non-uniformly-smooth field).
        Scope kept honest: on a UNIFORMLY-smooth field there is no concentration to exploit, so it ~ties uniform --
        the win is quality MOVED to where the field needs it, not free quality. Returns the anchor x-positions."""
        from holographic.caching_and_storage.holographic_adaptive_cache import adaptive_anchors
        return adaptive_anchors(x, y, n, floor=floor, power=power)

    def reconstruct_from_anchors(self, x, anchor_x, y):
        """Piecewise-linear reconstruction of field `y` (sampled at `x`) from its values at `anchor_x` (CACHE-3) --
        the cache read paired with adaptive_anchors: sample at the anchors, then interpolate between them."""
        from holographic.caching_and_storage.holographic_adaptive_cache import reconstruct_from_anchors
        return reconstruct_from_anchors(x, anchor_x, y)

    def robust_accumulate(self, samples, schedule="harmonic", alpha=0.2, clamp_k=None, exact=False):
        """Average noisy estimates of one quantity robustly, for the engine's averaging paths (consolidation over
        a growing store, forest vote-averaging). schedule='harmonic' uses 1/n weights (ACCUM-2: converges, best
        for a STATIONARY target; 'ema' tracks a DRIFTING target but plateaus; 'mean' is the plain mean). clamp_k
        (ACCUM-3), if set, winsorizes outlier samples to clamp_k robust-scales from the median first, so one
        firefly can't dominate -- measured ~100x lower error under outliers, with no loss on clean data."""
        from holographic.misc.holographic_accumulate import robust_accumulate
        # exact=True reduces through reduce_sum_exact -- integer accumulation, so the
        # SAME samples in a different ORDER give the SAME sum. Float addition is
        # not associative; this is the determinism guarantee, and it was
        # unreachable from the mind.
        return robust_accumulate(samples, schedule=schedule, alpha=alpha, clamp_k=clamp_k, exact=exact)

    def capacity_report(self, alpha=0.05, loads=(64, 256, 1024), n_floor=800, n_fa=800):
        """Where this store sits relative to the noise-wins CLIFF (Plate's HRR capacity theory), AND whether
        the calibrated false-alarm rate holds as the store GROWS (Cranmer's coverage-vs-LOAD -- the question
        Tier 0 left open by validating coverage only at a FIXED store). One report, two readings of the same
        geometry; the capacity complement to `calibration_report` (which checks coverage at the current store)
        and to `resolution_profile`.

        CAPACITY (the operating point, read off the actual prototype-row geometry where HRR theory applies):
          * `dprime` = (genuine-match cosine - noise-floor mean) / noise-floor std -- the SNR, in noise-sigmas,
            that a real match sits above what random crosstalk produces. Large is comfortably above the cliff;
            near 0 is AT it (noise wins).
          * `floor_mean` vs `hrr_floor_bound`: the measured noise floor (a random query's best cosine to the N
            rows) against the extreme-value bound sqrt(2 ln N / D) for N atoms in D dims -- the validation that
            the store's geometry behaves as the capacity theory predicts.
          * `headroom`: `n_cliff` = exp(D * match^2 / 2) is the store size at which the rising floor reaches the
            match level; `headroom` = n_cliff / N_now -- how many times the store could grow before noise wins
            (huge in high D for well-separated items, the whole point of distributed codes).

        COVERAGE vs LOAD: for each size in `loads`, build a random codebook of that many atoms in this mind's D,
        fit the procedure-matched recall null on it, and measure the false-alarm rate (fraction of pure-noise
        queries with p <= alpha). It should stay ~alpha as N grows -- the null re-fits to the rising floor, so
        the look-elsewhere discipline stays calibrated under load; materially above alpha at large N would mean
        the null is under-sampling the bigger store. Returns operating point, theory comparison, headroom, and
        per-load coverage. Deterministic (seeded by this mind)."""
        from holographic.agents_and_reasoning.holographic_honesty import RecallNull
        D = self.dim
        mat = self.memory.live._stack()[1]
        N = int(getattr(mat, "shape", (0,))[0])
        if N == 0:
            return {"n_prototypes": 0, "dim": D}
        rng = np.random.default_rng(self.seed)
        unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
        floor = np.array([float(np.max(mat @ unit(rng.standard_normal(D)))) for _ in range(n_floor)])
        mu, sd = float(floor.mean()), float(floor.std())
        match = float(np.mean(self._match_scores()))
        dprime = (match - mu) / (sd + 1e-12)
        hrr_bound = float(np.sqrt(2.0 * np.log(max(2, N)) / D))            # expected max of N cosines in D dims
        n_cliff = float(np.exp(min(700.0, D * match * match / 2.0)))       # cap exponent to avoid float overflow
        headroom = n_cliff / N
        coverage = {}
        for nload in loads:                                               # does coverage hold as the store grows?
            cb = np.stack([unit(rng.standard_normal(D)) for _ in range(int(nload))])
            rn = RecallNull().fit(cb, n_null=1500, seed=self.seed)        # procedure-matched null for this size
            qp = np.array([rn.pvalue(float(np.max(cb @ unit(rng.standard_normal(D))))) for _ in range(n_fa)])
            coverage[int(nload)] = float(np.mean(qp <= alpha))
        return {"n_prototypes": N, "dim": D, "match": match, "floor_mean": mu, "floor_std": sd,
                "dprime": dprime, "hrr_floor_bound": hrr_bound, "n_cliff": n_cliff, "headroom": headroom,
                "headroom_log10": float(np.log10(headroom)) if headroom > 0 else float("-inf"),
                "alpha": alpha, "coverage_vs_load": coverage}

    def conformance_report(self, dim=64, seed=0):
        """Run the ISA conformance suite (ISA-2): check every production base instruction against its
        definitional reference implementation, per the contract in ISA.md. Returns {op: {'passed', 'class', 
        'max_diff'}} where class is 'TOL' (a continuous output, conformant within numeric tolerance) or 'EXACT'
        (a decision / exact reindex, conformant bit-for-bit). The kernel is conformant iff every op passes -- and
        this is what makes a vectorized op safe to adopt: it is 'conformant' iff it passes here. The bind_batch
        class (a value-conformant change that flips a decision) is caught because decisions are pinned
        separately and exactly (see test_isa_conformance.py)."""
        from holographic.misc.holographic_reference import run_conformance
        return run_conformance(dim=dim, seed=seed)

    def calibration_report(self, n=2000, alphas=(0.01, 0.05, 0.1, 0.2), seed=12345):
        """Validate that the false-alarm probabilities recognize() and recall_calibrated() report are
        actually CALIBRATED. Draw `n` random unit vectors -- pure noise, matching nothing by construction --
        score each against the mind's own prototypes (the recognize path) and its individual store (the
        recall path), and report the empirical false-alarm RATE: the fraction whose p-value lands at or below
        each alpha. A calibrated detector fires on noise at rate ~= alpha; materially above alpha is
        anti-conservative (too many false matches), materially below is conservative. This is the radio-SETI
        / particle-physics coverage check (does thresholding at alpha hold the false-alarm rate at alpha?),
        run on the mind's own geometry -- and the validation that the procedure-matched recall null is not
        the anti-conservative under-estimate the earlier sampled null was."""
        rng = np.random.default_rng(seed)
        pnull = self._recognition_null()
        rnull = self._recall_null()
        proto = self.memory.live._stack()[1] if pnull is not None else None
        p_proto, p_indiv = [], []
        for _ in range(n):
            v = rng.standard_normal(self.dim); v /= np.linalg.norm(v) + 1e-12
            if pnull is not None and proto is not None and proto.shape[0]:
                p_proto.append(pnull.pvalue(float((proto @ v).max())))
            if rnull is not None:
                p_indiv.append(rnull.pvalue(float(self._recall.recall(v)[1])))
        rates = lambda ps: {a: (float(np.mean(np.asarray(ps) <= a)) if ps else float("nan")) for a in alphas}
        return {"n": int(n), "alphas": list(alphas),
                "prototype_false_alarm": rates(p_proto),       # recognize() path: should track alpha
                "individual_false_alarm": rates(p_indiv)}       # recall_calibrated() path: should track alpha

    def federation_report(self, target_items=None, threshold=0.90, n_vals=256, seed=0):
        """A federation / conservation diagnostic -- Path D's 'as above, so below' law as a callable readout,
        the federation-aware companion to `capacity_report` (which charts a SINGLE vector's noise-wins cliff).
        One D-vector holds only ~0.05-0.1 x D symbols at `threshold` cleanup-gated recall (the exact figure
        depends on the threshold); that budget is CONSERVED, so capacity comes from FEDERATING across shards,
        not from packing harder. Measured on the mind's own dimension and kernel (delegating to `storage_array`
        / HoloArray):
          * `per_vector_budget` -- the largest single-shard load whose recall still clears `threshold`;
          * `federated` -- a spot check that K aligned shards hold ~K x that budget at the same recall;
          * `conservation_ratio` -- partitioning the dimension in half holds total capacity (a half-D vector
            holds ~half the budget, so two of them tie one full vector): 2 x budget(D/2) / budget(D) ~ 1, the
            block-federation finding that federation buys capacity from more DIMENSIONS, not for free;
          * `recommended_shards` -- ceil(target_items / per_vector_budget), when `target_items` is given.
        Honest scope: this is the DISCRETE-symbol (cleanup-gated) budget ~0.1 x D; continuous compute with no
        cleanup is the lower ~0.02 x D regime (see `distributed_forward`), and federation buys fidelity and
        capacity, not fewer FLOPs."""
        from holographic.misc.holographic_array import HoloArray
        fracs = (0.05, 0.08, 0.10, 0.12, 0.15)

        def budget_at(dim):
            """Largest single-shard load (at this dimension) whose recall clears the threshold."""
            best = 0
            for load in (int(f * dim) for f in fracs):
                if load < 1:
                    continue
                arr = HoloArray(dim, seed=seed, n_parity=0, add_threshold=0.0, n_vals=n_vals)
                rng = np.random.default_rng(seed)
                for _ in range(load):
                    arr.add(int(rng.integers(0, n_vals)))
                if arr.accuracy() >= threshold:
                    best = load
            return best

        D = self.dim
        budget = max(budget_at(D), 1)

        # federated spot check: K aligned shards (each at the per-vector budget) hold ~K x budget at threshold
        K = 4
        arr = self.storage_array(n_parity=0, add_threshold=0.0, n_vals=n_vals)
        rng = np.random.default_rng(seed + 1)
        for k in range(K):
            if k > 0:
                arr._spin_up()
            for _ in range(budget):
                arr.add(int(rng.integers(0, n_vals)))
        fed_acc = float(arr.accuracy())

        # conservation: a half-dimension vector holds ~half the budget (so partitioning conserves total capacity)
        conservation_ratio = 2.0 * max(budget_at(D // 2), 1) / budget

        out = {"dim": D, "threshold": threshold,
               "per_vector_budget": budget, "per_vector_fraction": budget / D,
               "federated": {"shards": K, "stored": K * budget, "recall": fed_acc},
               "conservation_ratio": float(conservation_ratio)}
        if target_items is not None:
            out["target_items"] = int(target_items)
            out["recommended_shards"] = int(np.ceil(target_items / budget))
        return out

    # -- one decision brain, on the same substrate -------------------------
    def actions(self, names, robust_returns=False, value_backend="table"):
        """Declare the creature brain's action set. `robust_returns=True` (D2, opt-in) winsorises outlier rewards
        in each prototype's running-mean value: a fluke reward (a jackpot, a sensor glitch) is clamped to a few
        robust-scales before it folds in, so it cannot swing the value estimate -- measured ~3x lower value error
        under outlier rewards, no cost on clean data. Off by default (the plain running average).

        `value_backend` picks the brain's value/policy representation: 'table' (default, the prototype memory),
        'holo' (the two-bundle hypervector policy), or 'routed' (the hypervector policy with the routing fabric
        pushing the capacity cliff back). With 'holo'/'routed' the whole brain runs on a fixed-size, savable,
        composable hypervector policy -- so anywhere the creature is used in the holographic space, the
        holographic creature can be used instead (decide/reinforce are unchanged)."""
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000,
                                      maintain=self.maintain, robust_returns=robust_returns,
                                      value_backend=value_backend)
        return self

    def use_holographic_brain(self, routed=False):
        """Swap the (already-declared) creature brain to a holographic backend in place -- 'routed' for the
        cliff-pushed-back routing variant, else the plain two-bundle policy. Lets existing code that calls
        decide/reinforce run on a hypervector policy without re-plumbing. (Re-declares a fresh brain on the
        same action set; any table-mode learning is not carried over.)"""
        if not self._actions:
            raise RuntimeError("declare actions() before switching the brain backend")
        return self.actions(self._actions, value_backend=("routed" if routed else "holo"))

    def decide(self, state, explore=False, epsilon=None, modality=None,
               senses=None, avoid=("danger", "wall"), explore_if_unrecognized=None):
        """Decide an action. `senses`/`avoid` pass straight through to the
        brain's built-in safety reflexes (HolographicMind.decide): hand over
        the current senses dict and moves into seen dangers or walls are
        vetoed below the value estimate -- the unified brain gets the same
        measured safety every other caller of the model gets.

        `explore_if_unrecognized=alpha` carries the honesty layer from perception
        to ACTION: if the current state is noise-level against the brain's own
        experience (calibrated false-alarm p > alpha, see decide_confidence), the
        value estimate is built on nothing, so take a safe random move among the
        allowed actions instead of committing to an unreliable greedy pick -- the
        agent KNOWING when it is guessing. Off by default (None); the calibrated
        threshold replaces the brain's hand-set absolute `blind_floor` cosine."""
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        sv = self.perceive(state, modality)
        if explore_if_unrecognized is not None:
            null = self._brain_null()
            if null is not None:
                sup = max((self._brain.value(sv, a)[1] for a in range(len(self._actions))), default=0.0)
                if float(null.pvalue(float(sup))) > explore_if_unrecognized:
                    epsilon = 1.0                    # unrecognized: the value estimate is noise -> safe random
        a = self._brain.decide(sv, explore=explore, epsilon=epsilon, senses=senses, avoid=avoid)
        return self._actions[a]

    def _brain_null(self, n_null=800):
        """The noise floor for the creature brain's recognition of a STATE -- the action-side analogue of
        _recall_null, and PROCEDURE-MATCHED the same way: draw random unit states, run them through the
        brain's own value() (which projects into the brain's basis exactly as a real decision does), and take
        the distribution of the best support any action reaches. Calibrated by construction (the null IS the
        support noise produces under the real value path); uses value() as a black box, reaching into no
        internals. Cached on the brain's prototype count. Returns None when the brain has learned nothing."""
        if self._brain is None:
            return None
        nproto = int(sum(len(self._brain._unit[a]) for a in range(len(self._actions))))
        if nproto == 0:
            return None
        cache = getattr(self, "_brain_null_cache", None)
        if cache is None or cache[0] != nproto:
            rng = np.random.default_rng(self.seed)
            scores = np.empty(n_null)
            for i in range(n_null):
                s = rng.standard_normal(self.dim); s /= np.linalg.norm(s) + 1e-12
                scores[i] = max((self._brain.value(s, a)[1] for a in range(len(self._actions))), default=0.0)
            from holographic.agents_and_reasoning.holographic_honesty import RecallNull
            rn = RecallNull(); rn.null = np.sort(scores)
            self._brain_null_cache = (nproto, rn)
        return self._brain_null_cache[1]

    def decide_confidence(self, state, modality=None, explore=False, epsilon=None,
                          senses=None, avoid=("danger", "wall")):
        """Decide an action AND report a CALIBRATED confidence in it: (action, pvalue). The p-value is the
        false-alarm probability that the state is no better matched to the brain's experience than pure noise
        -- p small means the brain has genuinely been somewhere like here and the value estimate can be
        trusted; p large means it is in unfamiliar territory and effectively guessing. This is recognize()'s
        honesty applied to the decision brain: the same RecallNull machinery, over the brain's experienced
        states instead of the perceptual prototypes. The action returned is exactly decide()'s."""
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        sv = self.perceive(state, modality)
        a = self._brain.decide(sv, explore=explore, epsilon=epsilon, senses=senses, avoid=avoid)
        null = self._brain_null()
        if null is None:
            return self._actions[a], float("nan")
        sup = max((self._brain.value(sv, a2)[1] for a2 in range(len(self._actions))), default=0.0)
        return self._actions[a], float(null.pvalue(float(sup)))

    def reinforce(self, state, action, reward, modality=None):
        """Reinforcement-learning update: teach the decision brain that taking `action` in `state` earned `reward`.
        Perceives `state` into a hypervector (modality inferred when None), then records the (state, action, reward)
        experience so future decide()/act() calls prefer higher-reward actions in similar states. `action` must be one
        the mind already knows (see self._actions). Returns self (chainable), so you can stream experiences."""
        s = self.perceive(state, modality)
        self._brain.remember([s], [self._actions.index(action)], [float(reward)])
        return self

    # -- generation: predict the next symbol over the same space ------------
    def learn_sequence(self, data, n=6, hierarchical=True, modality="text", name=None):
        """Learn to continue a sequence.

        Two engines, picked by `hierarchical`:

        * The fractal coder (default): discover a chunk schema by compression, then predict by
          cross-level backoff -- emit the longest chunk a level is confident about, else descend
          a level and spell it out. Measured against the flat n-gram on Austen, it cut bits/char
          from 2.085 to 1.829 and the stored model from ~218k context entries to ~58k (3.8x
          smaller), at roughly tied coherence (0.96 vs 0.98 real words). Generation is the
          traversal-shaped operation where the multi-scale substrate earns its keep -- unlike
          classification, where a tree REGRESSED and the flat scan stayed best.

        * The flat holographic n-gram (`hierarchical=False`): the original engine, kept because
          it exposes `next_symbol` and an exact context key, and because the boundary between
          where the substrate helps and where it doesn't is measured here, not assumed.

        Two consolidations, both backward compatible:

        * `modality` passes through to the fractal coder, so the mind can learn to
          continue CODE, not just prose -- the same compress-by-merging schema was
          measured to discover code structure from scratch (held-out bits/char 2.98
          -> 2.28 on this project's own source, with `def __init__` and indentation
          idioms among the unlabeled emergent chunks).
        * `name` lets the mind hold MANY sequence schemas at once. Unnamed calls keep
          the old single-slot behaviour (each call replaces); named calls accumulate,
          and generate() with no name picks the schema by the compression gate -- the
          one routing primitive used everywhere else in the stack. That is
          content-level self-discovery, needed exactly where TYPE-level inference
          goes blind: code and prose are both `str`."""
        if hierarchical:
            from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
            gen, kind = SchemaGenerator(modality=modality).fit(data), "hierarchical"
        else:
            from holographic.misc.holographic_text import HolographicNGram
            gen = HolographicNGram(dim=self.dim, n=n, seed=0).fit(data)
            kind = "flat"
        key = name if name is not None else "default"
        if not hasattr(self, "_gens"):
            self._gens = {}
        self._gens[key] = {"gen": gen, "kind": kind, "modality": modality}
        self._gen, self._gen_kind = gen, kind        # most-recent alias (compat)
        return self

    def _pick_gen(self, name=None, seed_text=""):
        """Resolve which sequence schema a call means. Named -> that one. One schema
        -> it. Several and unnamed -> route the SEED by the compression gate: whoever
        compresses the seed best is the schema that understands it. The honest
        boundary: only hierarchical schemas expose bits_per_char, so flat engines
        never compete in the gate -- name them explicitly."""
        gens = getattr(self, "_gens", {})
        if not gens:
            raise RuntimeError("nothing learned to continue -- call learn_sequence() first")
        if name is not None:
            if name not in gens:
                raise KeyError(f"no sequence schema named {name!r} -- have {sorted(gens)}")
            return gens[name]
        if len(gens) == 1:
            return next(iter(gens.values()))
        from holographic.simulation_and_physics.holographic_schema import compression_gate
        gated = {k: g["gen"] for k, g in gens.items() if g["kind"] == "hierarchical"}
        if not gated or not seed_text:
            raise RuntimeError("several schemas are loaded -- name one, or give a seed "
                               "the gate can route (flat engines must be named)")
        return gens[compression_gate(seed_text, gated)[0][1]]

    def next_symbol(self, context, name=None):
        """Predict the next symbol (character) that should follow `context`, using a learned FLAT n-gram sequence
        schema. `name` selects among several learned schemas (defaults to the one matching the context). Returns the
        single most likely next character. Requires a flat engine -- train one with
        learn_sequence(text, hierarchical=False); raises RuntimeError if the selected schema is the hierarchical
        (fractal) kind, which decodes its own way. For whole continuations rather than one step, use generate()."""
        g = self._pick_gen(name, context)
        if g["kind"] != "flat":
            raise RuntimeError("next_symbol needs the flat engine: learn_sequence(text, hierarchical=False)")
        return g["gen"].next_char(context)

    def generate(self, seed_text, length=160, temperature=0.5, name=None, top_p=1.0):
        """Continue text from the chosen sequence schema. top_p<1.0 requests nucleus
        decoding; it is forwarded only to flat n-gram generators that support it (the
        hierarchical schema generator decodes its own way), so the argument is safe and
        backward-compatible everywhere."""
        gen = self._pick_gen(name, seed_text)["gen"]
        try:
            return gen.generate(seed_text, length, temperature, top_p=top_p)
        except TypeError:
            return gen.generate(seed_text, length, temperature)   # generator without top_p


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p02_fit_deterministic", "_UnifiedPart02")
    print("holographic_unified_p02_fit_deterministic selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
