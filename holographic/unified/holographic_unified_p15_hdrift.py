"""Part 15 of UnifiedMind's faculty surface -- HDRIFT: generative models as moment hypervectors.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which remains the only import path anyone uses.

WHY THIS PART EXISTS
--------------------
The HDRIFT arc (plan H0-H1) ships a generative engine whose model IS d+1 moment hypervectors:
training is one encoding pass, sampling is particle drift read off the vectors by dot products,
and the model algebra (compose by +, ablate by -, condition by unbind, transport by bind) is the
functionality no per-dataset-trained generator has. Rule-0 audit on record: 'novelty of generated
samples', 'combine two trained models', 'train on images and generate more' all returned fallbacks
-- the license to build. Two wiring promotions ride along: `write_wav` existed in holographic_audio
but never reached the mind (find_capability('write a wav audio file') returned file_write -- a pure
gap with working code behind it), and the auto-scaling integration (`drift_scale`) routes HDRIFT's
knobs through the EXISTING mind.auto_scale rather than growing a private tuner.

Every method DELEGATES; none reimplements. Each is a new name: no existing faculty's behaviour
changes and no emitted bytes flip.
"""

import numpy as np

from holographic.unified import check_part


class _UnifiedPart15:

    # ------------------------------------------------------------------ HDRIFT: train / generate

    def drift_train(self, points, labels=None, dim=1024, bandwidth=None, force=False, bounds=None):
        """TRAIN a holographic drift generative model on raw points: one encoding pass builds the
        kernel mean embedding + first-moment bundles (labels given -> every class packed into ONE
        vector set under unitary roles). Bandwidth is probed FROM THE DATA and a universally
        collapsing dataset is REFUSED, not served as a mean-generator (force=True overrides).
        Returns a saveable DriftModel (.save / mind.drift_load). See holographic_hdrift.build_drift_model."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.build_drift_model(points, labels=labels, dim=dim, seed=self.seed,
                                     bandwidth=bandwidth, force=force, bounds=bounds)

    def drift_load(self, path):
        """Load a saved DriftModel (moments + encoder recipe; the codebook regenerates from the seed,
        so only numbers ship). See holographic_hdrift.DriftModel.load."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.DriftModel.load(path)

    def drift_generate(self, model, n=64, condition=None, steps=60, repel=0.5, seed=None,
                       coupling="rownorm"):
        """SAMPLE a drift model: particles attract to the data field and repel from their OWN batch
        field (the corrective for the measured attraction-only memorisation), annealed noise to zero.
        `condition=` unbinds one label's field from a packed model -- conditional generation with no
        conditioning machinery; conditioned starts are importance-seeded so particles never traverse
        crosstalk dead zones. `coupling='sinkhorn'` (H0.4, measured): moment-native two-sided
        balancing that prevents low-temperature mode collapse (worst-mode share 0.236 +/- 0.020 vs
        rownorm 0.172 +/- 0.059 over 6 seeds, no collapse seeds, novelty_min 3x) at 2n extra dot
        products per step; default stays rownorm. Deterministic in seed.
        See holographic_hdrift.drift_sample."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.drift_sample(model, n=n, steps=steps, repel=repel, coupling=coupling,
                                seed=self.seed if seed is None else seed, condition=condition)

    # ------------------------------------------------------------------ HDRIFT: the model algebra

    def drift_compose(self, a, b):
        """COMBINE two drift models trained separately, never co-trained: moment vectors ADD
        (evidence-weighted -- sums carry n). The models must share one encoder space.
        See holographic_hdrift.drift_compose."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.drift_compose(a, b)

    def drift_ablate(self, a, b):
        """REMOVE model b's contribution from model a by subtraction -- unlearning / a negative prompt
        with no retraining. Exact when b's data is a subset of a's; an approximation otherwise, and a
        negative region reads as near-zero density (refusal), not anti-matter.
        See holographic_hdrift.drift_ablate."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.drift_ablate(a, b)

    def drift_transport(self, model, delta):
        """MOVE a whole trained distribution by `delta` without touching data: FPE shift-is-a-bind on
        the moment bundles, with the first-moment cross-term (nu' = shift(nu) + delta*shift(mu)) that
        the naive shift drops. See holographic_hdrift.drift_transport."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.drift_transport(model, delta)

    # ------------------------------------------------------------------ H0.1: the gate

    def generation_audit(self, samples, train, k_modes=None):
        """NOVELTY + COVERAGE of generated samples against their training set, in one report --
        memorisation manifests as success (perfect samples), so nothing generated should ship without
        this attached. novelty ~0 = memorised (nearest-training distance in units of the training
        set's own NN scale); coverage = fraction of k data modes some sample lands nearest to.
        See holographic_hdrift.generation_audit."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.generation_audit(samples, train, k_modes=k_modes, seed=self.seed)

    # ------------------------------------------------------------------ H1: media (images)

    def train_media_model(self, images, labels=None, k=8, dim=1024, fit_steps=150):
        """TRAIN A GENERATIVE MODEL ON IMAGES: each image -> k anisotropic splats (hand-derived-gradient
        Adam) -> one canonically-ordered point in splat-parameter space (dozens of dims, not thousands
        -- the curse-of-dimensionality answer) -> drift moments with probed bandwidth. Returns
        (model, meta); feed both to mind.generate_media. See holographic_hdrift.train_image_drift."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        return _hd.train_image_drift(images, labels=labels, k=k, dim=dim, seed=self.seed,
                                     fit_steps=fit_steps)

    def generate_media(self, model, meta, n=4, condition=None, steps=60, audit_train=None, seed=None):
        """GENERATE IMAGES from a trained media model: drift in splat space, render each particle,
        and ALWAYS attach the generation audit when audit_train is given -- a generation without its
        novelty/coverage numbers is the failure mode wearing a success costume.
        See holographic_hdrift.generate_images."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        out = _hd.generate_images(model, meta, n=n, seed=self.seed if seed is None else seed,
                                  condition=condition, steps=steps, audit_train=audit_train)
        return out

    # ------------------------------------------------------------------ auto-scaling integration

    def drift_autoscale(self, points, target_spread=0.9, max_rounds=6):
        """ROUTE HDRIFT's knobs through the mind's EXISTING auto_scale (no private tuner): eval_fn is
        the bandwidth prober's spread-fidelity score at the current (dim, bandwidth) operating point;
        auto_scale doubles the most responsive knob until the target is met or a WALL is named.
        Returns auto_scale's trajectory, every step carrying its probe. See holographic_hdrift.probe_bandwidth
        + holographic_scalinglaw (mind.auto_scale)."""
        import holographic.sampling_and_signal.holographic_hdrift as _hd
        pts = np.asarray(points, float)

        def _eval(knobs):
            rep = _hd.probe_bandwidth(pts, dim=int(knobs["dim"]), seed=self.seed,
                                      candidates=(float(knobs["bandwidth"]),))
            s = rep["scores"][float(knobs["bandwidth"])]
            return abs(float(np.log(max(s, 1e-9))))            # 0 == perfect unit spread

        return self.auto_scale(_eval, {"dim": 1024, "bandwidth": 4.0},
                               target_error=abs(float(np.log(target_spread))), max_rounds=max_rounds)

    # ------------------------------------------------------------------ wiring promotion: audio out

    def write_wav(self, path, samples, rate):
        """Write float samples in [-1,1] to a 16-bit PCM WAV file -- the missing OUT half of read_wav
        (the function shipped in holographic_audio but never reached the mind; a generation pipeline
        that cannot emit audio is not a pipeline). See holographic_audio.write_wav."""
        import holographic.misc.holographic_audio as _au
        return _au.write_wav(path, samples, rate)


    # ------------------------------------------------------------------ VOID-1: the disciplined explorer

    def void_map(self, model, train, n_probes=512, n_null=24, alpha=0.05):
        """MAP WHERE A CORPUS HAS NOTHING: probe a drift model's support box and return the gated
        voids -- low-density points that survive the bootstrap null ('a finite sample of anything has
        pockets': sparsity the data's own noise explains is reported separately, never as void). The
        instrument builds its OWN sharpest-honest bandwidth: the sampler's smooth kernel measurably
        smears absence (gap read 56% of data density at the generation bandwidth, -6% at the
        instrument's). See holographic_voidexplore.void_map."""
        import holographic.agents_and_reasoning.holographic_voidexplore as _vx
        return _vx.void_map(model, train, n_probes=n_probes, seed=self.seed, n_null=n_null, alpha=alpha)

    def structured_voids(self, observations, min_count=2, max_candidates=64):
        """THE MENDELEEV MOVE on a discrete corpus: combinations the observed STRUCTURE licenses but
        the observed SET lacks (every pairwise slot co-occurrence seen >= min_count; only the full
        assembly is new). Gated by the anti-epicycle clause -- the grammar may vouch for unseen
        combinations only if its pairwise structure beats a slot-shuffle null; independent slots are
        REFUSED with the p-value, not enumerated. See holographic_voidexplore.structured_voids."""
        import holographic.agents_and_reasoning.holographic_voidexplore as _vx
        return _vx.structured_voids(observations, min_count=min_count,
                                    max_candidates=max_candidates, seed=self.seed)

    def transfer_voids(self, model_a, model_b, n=32, thresh=0.15):
        """PRESENT IN B, ABSENT IN A -- the cross-disciplinary warrant, strictly stronger than
        grammar validity: sample corpus B's drift model and keep points where A's density is low
        while B's is high (each scaled against its own on-support level). Not 'the structure allows
        it' but 'reality already contains it, elsewhere'. Both models must share one encoder space.
        See holographic_voidexplore.transfer_voids."""
        import holographic.agents_and_reasoning.holographic_voidexplore as _vx
        return _vx.transfer_voids(model_a, model_b, n=n, seed=self.seed, thresh=thresh)


    # ------------------------------------------------------------------ RESID-1: noise as unexplained data

    def residual_verdict(self, y, n_surrogates=64, min_seg=16, penalty=3.0,
                         scales=(4, 8, 16, 32, 64, 128)):
        """EXPLAIN, SUBTRACT, INTERROGATE WHAT REMAINS: decompose a series, subtract the explanation,
        and ask whether the explanation removed all temporal dependence IN BOTH MOMENTS: the LEVEL
        channel (autocorrelation) and the SCALE channel (squared-residual dependence -- volatility
        clustering; the single-channel version FALSELY REFUSED an ARCH residual at p=0.39 while its
        squared series measured 43x the level stat). Null: iid_shuffle (marginal preserved EXACTLY;
        order destroyed in both moments at once); 'structured' if EITHER channel fires, the firing
        channel named, with a block-scale containment PROFILE, else 'irreducible' ('refusal is a
        result'; an efficient market's residual SHOULD read irreducible). FURTHER KEPT NEGATIVES: the
        verdict is conditional on the explainer's capacity (smooth deterministic structure is
        ABSORBED into segment laws), and the first design demanded AAFT+block nulls simultaneously
        -- conflated claims: block surrogates CONTAIN short structure, AAFT preserves the spectrum
        that linear dependence IS. One claim, one matched null.
        See holographic_residualvoid.residual_verdict."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.residual_verdict(y, n_surrogates=n_surrogates, seed=self.seed,
                                    min_seg=min_seg, penalty=penalty, scales=scales)

    def support_gauge(self, y, embed=4, train_window=256, hop=8, dim=1024, n_null=16):
        """HAVE I SEEN A STATE LIKE THIS? A CAUSAL out-of-support monitor: at each step, drift moments
        are built from the TRAILING window only and z(now) is read against the history's own scale,
        bootstrap-gated into inside / sparse / void. Predicts NOTHING about a void's contents -- it
        reports that you have ENTERED one, the model-validity claim that does not decay when others
        hold it. THE VOID CLOSES AS IT IS OBSERVED: the trailing window absorbs a new regime and the
        gauge recovers -- adaptation is the contract. See holographic_residualvoid.support_gauge."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.support_gauge(y, embed=embed, train_window=train_window, hop=hop, dim=dim,
                                 seed=self.seed, n_null=n_null)

    def hidden_drivers(self, panel, n_surrogates=48, min_seg=16, penalty=3.0):
        """THE PUPPET STRINGS: explain every series in a panel separately, then test whether their
        RESIDUALS share a common factor beyond independently-surrogated panels (AAFT per residual --
        the null that destroys exactly the co-movement claim). A passing factor is an influence
        outside every single-series explanation: news, a common counterparty, an exploit. Refused
        ('independent') when the unexplained parts do not co-move. Recovery is bounded by what
        survives explanation -- the EXISTENCE verdict is the strong claim, the factor estimate its
        surviving shadow. See holographic_residualvoid.hidden_drivers."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.hidden_drivers(panel, n_surrogates=n_surrogates, seed=self.seed,
                                  min_seg=min_seg, penalty=penalty)


    def panel_gauge(self, panel, corr_window=60, train_window=240, hop=20, dim=1024, n_null=12,
                    panel_bandwidth=3.0, state_map="corr", tail_q=0.90):
        """HAVE THE RELATIONSHIPS EVER LOOKED LIKE THIS? Joint-panel out-of-support monitor: the
        state is the Fisher-z upper triangle of the trailing correlation matrix, gauged causally
        against the history of such states -- the void support_gauge cannot see (a correlation
        crisis puts the DEPENDENCE structure outside all history while every marginal sleeps; the
        selftest plants exactly that and the marginal gauge stays silent). Three COSTUMES via
        state_map: 'corr' (Fisher-z correlations), 'leadlag' (the ANTISYMMETRIC lag-1 cross-corr:
        who moves first -- flips the corr costume is provably blind to, pinned), 'tail'
        (co-exceedance beyond each series' own tail_q quantile, arcsin-sqrt stabilised: do they
        crash together). A state outside the history's 10-90% ROBUST box is void BY GEOMETRY,
        unclipped -- min/max bounds let one straddling transition state grant deniability (kept
        negative).
        See holographic_residualvoid.panel_gauge."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.panel_gauge(panel, corr_window=corr_window, train_window=train_window, hop=hop,
                               dim=dim, seed=self.seed, n_null=n_null, panel_bandwidth=panel_bandwidth,
                               state_map=state_map, tail_q=tail_q)

    def residual_ladder(self, y, max_depth=3, n_surrogates=48, min_seg=16, penalty=3.0, ar_order=8):
        """CLIMB THE RESIDUAL: explain (piecewise), interrogate; while 'structured', apply the next
        grammar SELECTED BY CHANNEL -- level dependence gets the closed-form AR rung (subtract the
        prediction); scale-only dependence gets the vol-AR rung (DIVIDE by the fitted conditional
        envelope: a vol model explains the envelope, not the signs). Both rungs deterministic ridge,
        no learning loop. Re-interrogate. The terminal answer names WHICH grammar priced the remainder
        as noise, or admits none here did ('rungs-exhausted' -- the Mendeleev boundary: the tower
        cannot climb past the axioms it has). See holographic_residualvoid.residual_ladder."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.residual_ladder(y, max_depth=max_depth, n_surrogates=n_surrogates,
                                   seed=self.seed, min_seg=min_seg, penalty=penalty, ar_order=ar_order)

    def stream_watch(self, y, sentinel=None, embed=4, train_window=256, hop=8, dim=1024, n_null=12):
        """ONE TIMELINE: the regime sentinel's events and the support gauge's void events merged in
        the sentinel's own dialect ({at, kind, ...}) -- 'support-void' on entering territory no
        history covers, 'support-recovered' when the trailing window absorbs it (the closing is
        part of the story). Two monitors with two report formats is how an operator misses the
        morning both fire at once. See holographic_residualvoid.stream_watch."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.stream_watch(y, sentinel=sentinel, embed=embed, train_window=train_window,
                                hop=hop, dim=dim, seed=self.seed, n_null=n_null)


    def market_residual_report(self, n_surrogates=64, max_n=1500):
        """RUN THE RESIDUAL LADDER ON THE CHECKED-IN MARKET DATA (DAI/WETH 1m, SOL/USDT 1h returns
        + levels, SOL tick moves) and report which grammar terminates each stream. First run
        reproduced the STYLIZED FACTS with no market knowledge in the code: 1h returns level-clean
        but scale-structured (volatility clustering, vol rung terminates -- Engle's finding read
        off the tower); tick moves fire the AR rung with a NEGATIVE lag-1 coefficient (~-0.21, the
        bid-ask bounce); tiny-n returns irreducible (EMH at low power, acknowledged); price levels
        consumed by ar(8) (a random walk is an AR fit's favourite meal).
        See holographic_residualvoid.market_residual_report."""
        import holographic.sampling_and_signal.holographic_residualvoid as _rv
        return _rv.market_residual_report(n_surrogates=n_surrogates, max_n=max_n, seed=self.seed)


    # ------------------------------------------------------------------ SCI-1: the period hunter

    def transit_search(self, times, values, min_period, max_period, n_periods=800, n_bins=64,
                       n_null=24, alpha=0.05):
        """FIND A PHASE-COHERENT PERIOD with the BOX-matched filter (Box Least Squares, Kovacs et
        al. 2002 -- deterministic, closed form; measured 6.3x more peak contrast than the sinusoid
        filter on a box near the detection floor, which is exactly where planets are lost). The
        verdict is judged against the PROCEDURE-MATCHED block-shuffle null (red noise survives,
        phase coherence dies -- an iid null flags red noise as planets and is reported, not used),
        the harmonic family is reported rather than hidden, and a surrogate budget whose p-floor
        cannot arithmetically pass REFUSES to pretend it ran.
        See holographic_transitbox.transit_search."""
        import holographic.sampling_and_signal.holographic_transitbox as _tb
        return _tb.transit_search(times, values, min_period, max_period, n_periods=n_periods,
                                  n_bins=n_bins, n_null=n_null, seed=self.seed, alpha=alpha)

    def transit_detection_floor(self, depths=(0.002, 0.004, 0.006, 0.010), period=173.0, dur=9,
                                n=2000, noise=0.002, n_seeds=4, n_null=24):
        """THE DETECTION-LIMIT CURVE, not a highlight reel: injected-box recovery fraction as a
        function of depth at fixed noise, each point carrying its per-transit SNR. The honest
        deliverable of any detector is where it STOPS working.
        See holographic_transitbox.detection_floor."""
        import holographic.sampling_and_signal.holographic_transitbox as _tb
        return _tb.detection_floor(depths=depths, period=period, dur=dur, n=n, noise=noise,
                                   n_seeds=n_seeds, n_null=n_null, seed=self.seed)


    def fold_subtract(self, times, values, period, n_bins=64, engine="median", dim=2048):
        """SUBTRACT THE PERIODIC PART of a series at a known period -- the fold rung as a verb.
        engine='median' (default): per-bin median template, outlier-proof; engine='vsa': the
        CircularEncoder kernel fold -- smooth, bin-free, UNEVEN/JITTERED SAMPLING NATIVE (shape
        from the bundle, amplitude from one closed-form projection; measured 92% box-power
        consumption on 40%-gapped stamps). Returns (residual, template).
        See holographic_transitbox.fold_subtract / vsa_fold."""
        import holographic.sampling_and_signal.holographic_transitbox as _tb
        return _tb.fold_subtract(times, values, period, n_bins=n_bins, engine=engine,
                                 dim=dim, seed=self.seed)


    # ------------------------------------------------------------------ SCI-2: the pulsar panel

    def hd_search(self, panel, positions, ar_order=8, n_null=32, alpha=0.05):
        """THE GRAVITATIONAL-WAVE-BACKGROUND PATTERN TEST (Hellings-Downs) on a panel of timing
        residuals: whiten each series (closed-form AR -- raw red-vs-red correlations are spurious,
        pinned), correlate every pair, and judge the pattern with TWO matched nulls -- AAFT per
        series (does ANY cross-correlation exist) and the SKY SCRAMBLE (permute positions against
        residuals: correlations survive, only geometry dies). Verdicts: 'hd-consistent',
        'correlated-not-sky-patterned' (the monopole/clock-error diagnosis -- a co-moving panel
        that ANY sky assignment explains equally), or 'independent'. Amplitude is a stated LOWER
        BOUND (per-series whitening attenuates shared signal); the certified quantity is the
        curve SHAPE. See holographic_pulsarpanel.hd_search."""
        import holographic.sampling_and_signal.holographic_pulsarpanel as _pp
        return _pp.hd_search(panel, positions, ar_order=ar_order, n_null=n_null,
                             seed=self.seed, alpha=alpha)

    def hd_panel_demo(self, k=12, n=1500, gw_amp=0.45, mode="hd"):
        """Synthetic pulsar-timing panel with PLANTED ground truth for the verdict experiment:
        per-pulsar red noise plus a cross-pulsar process with Hellings-Downs spatial covariance
        ('hd'), a constant-correlation MONOPOLE (the clock-error control, 'mono'), or nothing
        ('none'). Returns (panel, positions). The planted process is white in time on purpose so
        per-series whitening cannot eat it (the segmenter-eats-boxes lesson, pre-applied).
        See holographic_pulsarpanel.make_hd_panel."""
        import holographic.sampling_and_signal.holographic_pulsarpanel as _pp
        return _pp.make_hd_panel(k=k, n=n, gw_amp=gw_amp, seed=self.seed, mode=mode)


    # ------------------------------------------------------------------ SCI-3: the spectroscopist's bench

    def spectral_lines(self, x, y, catalog=None, min_snr=4.0, n_null=32, tol_frac=0.002):
        """FIND (and optionally IDENTIFY) lines in a measured spectrum: median continuum off,
        candidates gated against the max-hunting NOISE-ONLY bootstrap null (a permutation null
        contains its own lines -- kept negative), centers refined sub-bin. With a `catalog`
        ({name: rest_wavelength}), identification runs the cleanup discipline in scalar costume:
        nearest entry accepted only with a 2x margin over the runner-up -- between lines, ABSTAIN
        (an identification without a margin is a coin flip wearing a name).
        See holographic_spectralline.find_lines / identify_lines."""
        import holographic.sampling_and_signal.holographic_spectralline as _sp
        out = _sp.find_lines(x, y, min_snr=min_snr, n_null=n_null, seed=self.seed)
        if catalog is not None:
            out["identification"] = _sp.identify_lines(
                [l["center"] for l in out["lines"]], catalog, tol_frac=tol_frac)
        return out

    def redshift_verdict(self, centers, catalog, z_max=0.2, tol_frac=0.0015, n_null=48):
        """THE LE VERRIER MOVE ON A LINE LIST: one shared shift must explain EVERY measured line's
        displacement, judged against scrambled catalogs (same line density, pattern destroyed).
        The scan picks the ASSIGNMENT; the value is the median per-line z (the scan's best-z is
        the tolerance window's low edge -- kept negative). A single matched line is numerology;
        the verdict is agreement across the list, or 'no-consistent-shift' with the p-value.
        Velocity readout is classical c*z; the dedoppler faculty offers the relativistic form.
        See holographic_spectralline.redshift_verdict."""
        import holographic.sampling_and_signal.holographic_spectralline as _sp
        return _sp.redshift_verdict(centers, catalog, z_max=z_max, tol_frac=tol_frac,
                                    n_null=n_null, seed=self.seed)

    def fit_decay(self, t, y, n_boot=24):
        """FIT y = A exp(-lambda t) + C, closed form (counts, ringdowns, randomized-benchmarking
        fidelity curves): tail-median background + d^2-weighted log-linear LS -- the weights are
        the load-bearing choice (delta method: Var[log d] ~ 1/d^2; plain d-weights measured
        lambda 17% low, and a coordinate-descent background pass moved the WRONG way -- both kept
        negatives). Bootstrap CI, shuffle-null verdict gate, and the truncation flag with a
        bias-aware margin (a truncated record biases lambda HIGH -- the flag absorbs the very
        bias it reports). See holographic_spectralline.fit_decay."""
        import holographic.sampling_and_signal.holographic_spectralline as _sp
        return _sp.fit_decay(t, y, n_boot=n_boot, seed=self.seed)


    # ------------------------------------------------------------------ SCI-4: quantum statistics

    def level_statistics(self, levels, n_boot=400, trim_frac=0.1):
        """INTEGRABLE OR CHAOTIC, read off the spectrum alone: the consecutive-spacing RATIO
        statistic (Atas et al. 2013) -- NO unfolding, the local density cancels exactly, where a
        wrong unfolding manufactures or erases level repulsion. Classifies against the exact
        Poisson mean 2ln2-1 and the GOE/GUE surmises via a bootstrap CI, and REFUSES
        ('indeterminate', with the n that would decide) when the CI cannot separate the classes
        -- the p-floor lesson as a sample-size statement. Spectrum edges trimmed: universality
        lives in the bulk. See holographic_quantumstats.level_statistics."""
        import holographic.sampling_and_signal.holographic_quantumstats as _q
        return _q.level_statistics(levels, n_boot=n_boot, seed=self.seed, trim_frac=trim_frac)

    def chsh_verdict(self, a_setting, b_setting, a_out, b_out, n_null=200, n_boot=400):
        """THE BELL VERDICT on trial data, three gates and one alarm: the pairing-scramble null
        (B outcomes shuffled within each setting cell -- marginals survive, correlation dies)
        answers 'correlated at all?'; the bootstrap CI against the classical bound 2 answers
        'beyond every local hidden-variable model?'; and the TSIRELSON ALARM -- a CI clearing
        2*sqrt(2) reads 'suspect-instrument', because quantum mechanics itself stops there and
        data beyond it is accusing the apparatus (post-selection, pairing errors), not the
        theory. mind.chsh_demo plants all four regimes.
        See holographic_quantumstats.chsh_verdict."""
        import holographic.sampling_and_signal.holographic_quantumstats as _q
        return _q.chsh_verdict(a_setting, b_setting, a_out, b_out, n_null=n_null,
                               n_boot=n_boot, seed=self.seed)

    def chsh_demo(self, n=4000, kind="quantum"):
        """Planted CHSH trials for the verdict experiment: 'quantum' (singlet statistics at the
        optimal angles), 'classical' (an explicit local hidden-variable model -- S<=2 by
        construction; if the verdict calls THIS nonclassical, the instrument, not Bell, is
        wrong), 'independent' (coins), 'broken' (sign-aware post-selection -- the selection
        loophole made concrete, pushing S past Tsirelson so the alarm can be exercised).
        See holographic_quantumstats.make_chsh_trials."""
        import holographic.sampling_and_signal.holographic_quantumstats as _q
        return _q.make_chsh_trials(n=n, kind=kind, seed=self.seed)


    # ------------------------------------------------------------------ SCI-5: the front door

    def science_report(self, data, kind, **kw):
        """ONE FRONT DOOR for the science instruments: route `data` (dict of named fields, or a
        tuple in declaration order) to the matching instrument by an EXPLICIT `kind` -- one of
        light_curve / pulsar_panel / spectrum / decay / levels / chsh / series -- and return the
        uniform report {'kind','verdict','why','result'}. An unknown kind raises WITH the list:
        the door never guesses, because the wrong instrument returns a confident nonsense
        verdict. Every route inherits its instrument's refusals verbatim. The faculty-to-
        literature-ancestor map, with citations, is docs/SCIENCE_INSTRUMENTS.md.
        See holographic_sciencereport.science_report."""
        import holographic.sampling_and_signal.holographic_sciencereport as _sr
        return _sr.science_report(data, kind, seed=self.seed, **kw)


    # ------------------------------------------------------------------ HDRIFT Phase 2: audio

    def train_audio_drift(self, clips, rate, n_tones=2, dim=2048):
        """TRAIN a drift model on audio clips, where the abstention ladder IS the adapter: a clip
        maps to (freq, amp) tone parameters when fit_multitone's r2 gate passes (frequency-sorted
        -- phase is gauge and deliberately dropped), to a log-band spectral envelope when it is a
        STATIONARY texture, and is refused when it is neither (a chirp: one point cannot honestly
        describe it in v1). A corpus must be ONE space -- a mixed corpus refuses with the mode
        counts. Returns (model, meta) or the refusal dict.
        See holographic_driftaudio.train_audio_drift."""
        import holographic.sampling_and_signal.holographic_driftaudio as _da
        return _da.train_audio_drift(clips, rate, n_tones=n_tones, dim=dim, seed=self.seed)

    def generate_audio(self, model, meta, n=4, steps=60, coupling="rownorm"):
        """GENERATE audio from a trained drift model: drift in the adapter's space, resynthesize
        deterministically (exact additive sine for tones -- store the formula, the HRNN move;
        seeded envelope-shaped noise for textures), and ALWAYS attach the audit plus the
        nearest-training band-spectral distance -- a generation without its numbers does not
        return. Write results with mind.write_wav.
        See holographic_driftaudio.generate_audio."""
        import holographic.sampling_and_signal.holographic_driftaudio as _da
        return _da.generate_audio(model, meta, n=n, seed=self.seed, steps=steps, coupling=coupling)


    # ------------------------------------------------------------------ HDRIFT Phase 3: video

    def train_video_drift(self, clips, k=2, dim=2048):
        """TRAIN a drift model on short clips (stacks of frames): each clip becomes a
        keyframe-PAIR point [start splats, end-minus-start] -- motion is the JOINT STRUCTURE
        between keyframes (the thing H1.4 proved the model preserves and marginals scramble),
        with end splats re-matched to start splats by nearest centre so the delta describes
        motion, not a relabelling. Single-frame clips refuse the corpus with the count.
        See holographic_driftvideo.train_video_drift."""
        import holographic.sampling_and_signal.holographic_driftvideo as _dv
        return _dv.train_video_drift(clips, k=k, dim=dim, seed=self.seed)

    def generate_video(self, model, meta, n=2, n_frames=8, steps=60, coupling="rownorm"):
        """GENERATE clips: drift a keyframe-pair point, interpolate splat params across
        n_frames, render every frame -- temporal coherence by construction and MEASURED anyway
        (per-clip max frame-to-frame RMS rides in the audit; a smoothness claim without its
        numbers is narrative). See holographic_driftvideo.generate_video."""
        import holographic.sampling_and_signal.holographic_driftvideo as _dv
        return _dv.generate_video(model, meta, n=n, n_frames=n_frames, seed=self.seed,
                                  steps=steps, coupling=coupling)

    def codec_atlas(self):
        """The compression family's SPEC SHEET (machine_map applied to codecs): every codec
        unit -- zlib/lzma, low-rank/tucker/tt, rate-distortion, pack_images, event codec,
        sequence-predictive, generator rung, cold storage -- with its real module+symbol,
        preconditions, pays-condition, and kept negatives. Static contracts; measure on YOUR
        data with codec_place. See holographic_codecatlas.codec_atlas."""
        from holographic.caching_and_storage.holographic_codecatlas import codec_atlas as _ca
        return _ca()

    def codec_place(self, x, max_error=None, try_lossy=None):
        """Which codec should this data use? MEASURES every applicable unit on x and returns a
        ranked table priced against the zlib baseline, with 'store raw' as a first-class row.
        Lossy units run ONLY when a max_error budget is stated (loss is never volunteered) and
        are gated by the error budget, never 99% energy. Refusal on incompressible data is the
        finding, not a failure. See holographic_codecatlas.codec_place."""
        from holographic.caching_and_storage.holographic_codecatlas import codec_place as _cp
        return _cp(x, max_error=max_error, try_lossy=try_lossy)

    def residual_encode(self, y, max_error=None, min_seg=64, penalty=3.0, max_terms=6):
        """Compress a 1-D signal as MODEL + CODED ERROR: decompose_piecewise fits per-segment
        laws, the residual is byte-plane-shuffled and entropy-coded. Exact by default
        (bit-identical decode, pinned); with max_error, near-lossless within the budget
        (measured 8.5x vs zlib). Self-refuses into mode='raw' when the model does not pay --
        a codec that cannot say 'store raw' is not honest.
        See holographic_residualcodec.residual_encode."""
        from holographic.sampling_and_signal.holographic_residualcodec import residual_encode as _re
        return _re(y, max_error=max_error, min_seg=min_seg, penalty=penalty,
                   max_terms=max_terms, mind=self)

    def residual_decode(self, blob):
        """Invert residual_encode: rebuild the prediction from the stored recipes, add the
        coded error back (exact mode bit-identical; quant mode within its stated budget;
        raw mode inflates the refused baseline). See holographic_residualcodec.residual_decode."""
        from holographic.sampling_and_signal.holographic_residualcodec import residual_decode as _rd
        return _rd(_as_blob(blob))

    def surprise_code(self, points, reference, fine_step, coarsen=128.0, dim=2048,
                      news_quantile=0.10):
        """Allocate bits by SURPRISE: points a reference corpus's drift model predicts get a
        coarse step, points in its void (the news, judged by z=<enc(x),mu> against the
        reference's own support scale) get fine_step -- same news fidelity as uniform-fine
        coding, measured 1.71x fewer bytes. Falls back to mode='uniform' when the news share
        sits at chance level (the split cannot pay). Lossy by design on the predicted mass --
        for bit-exact contracts use residual_encode or codec_place.
        See holographic_surprisecodec.surprise_code."""
        from holographic.sampling_and_signal.holographic_surprisecodec import surprise_code as _sc
        return _sc(points, reference, fine_step, coarsen=coarsen, dim=dim,
                   news_quantile=news_quantile, mind=self)

    def surprise_decode(self, blob):
        """Invert surprise_code: read the per-point news flags and dequantize each point at
        its own step (uniform mode: one step everywhere).
        See holographic_surprisecodec.surprise_decode."""
        from holographic.sampling_and_signal.holographic_surprisecodec import surprise_decode as _sd
        return _sd(_as_blob(blob))

    def distribution_encode(self, points, bits=6, dim=2048, n_audit=64, k_modes=2):
        """Compress a sample bank to its DISTRIBUTION: the drift model's d+1 moment
        hypervectors, quantized at 4/6/8 bits (measured 10.5x/21.5x vs zlib at coverage 1.0).
        Decode returns a DriftModel to sample from -- points LIKE the originals, never the
        originals; the report prices break_even_n and carries the post-quantization
        generation audit. Need exactness? codec_place / residual_encode.
        See holographic_distcodec.distribution_encode."""
        from holographic.sampling_and_signal.holographic_distcodec import distribution_encode as _de
        return _de(points, bits=bits, dim=dim, n_audit=n_audit, k_modes=k_modes, mind=self)

    def distribution_decode(self, blob):
        """Rebuild the DriftModel from a distribution blob (encoder from its numeric recipe,
        moments dequantized per-array); sample with mind.drift_generate(model, ...).
        See holographic_distcodec.distribution_decode."""
        from holographic.sampling_and_signal.holographic_distcodec import distribution_decode as _dd
        return _dd(_as_blob(blob))

    def store_procedural(self, y, tol=0.02):
        """Store a 1-D signal as its PROGRAM: generator-bank tier (constant-size blob --
        MEASURED 76x at n=4k and 310x at n=16k from the SAME bytes -- extendable past the
        data with a validity flag) or piecewise-recipe tier (11.4x, original length only);
        each tier VERIFIED pointwise at tol*amplitude before commit, refused with the
        measured errors and a route hint when both miss.
        See holographic_proccodec.store_procedural."""
        from holographic.sampling_and_signal.holographic_proccodec import store_procedural as _sp
        return _sp(y, tol=tol, mind=self)

    def regen_procedural(self, blob, n=None):
        """Regenerate a signal from its program blob: generator tier at ANY length
        (valid=False past 2x the verified window -- the reprojection-ghost bound); recipe
        tier at the original length only (extension on per-segment axes is refused, not
        extrapolated). See holographic_proccodec.regen_procedural."""
        from holographic.sampling_and_signal.holographic_proccodec import regen_procedural as _rp
        return _rp(_as_blob(blob), n=n)

    def mesh_encode(self, mesh, max_error, grid=12, try_base=True):
        """Compress a triangle mesh at a stated budget: vertices per-coordinate
        |err| <= max_error, connectivity BIT-EXACT, measured 2.5-2.7x vs zlib(raw). Always
        prices the base+displacement hypothesis against the fair uniform-quant coder and
        ships the smaller -- MEASURED NEGATIVE on record: explicit refs cost what the deltas
        save, so uniform wins on every mesh class tried (the module docstring carries the
        sweep). try_base=False skips pricing the known loser.
        See holographic_meshcodec.mesh_encode."""
        from holographic.mesh_and_geometry.holographic_meshcodec import mesh_encode as _me
        return _me(mesh, max_error, grid=grid, try_base=try_base, mind=self)

    def mesh_decode(self, blob):
        """Invert mesh_encode -> (vertices, faces): budget-honored vertices, bit-exact
        connectivity. See holographic_meshcodec.mesh_decode."""
        from holographic.mesh_and_geometry.holographic_meshcodec import mesh_decode as _md
        return _md(_as_blob(blob))




    def ablation_table(self, seeds=range(3)):
        """Run the VSA-load-bearing audit: for each subsystem, the dumbest honest non-holographic
        baseline on the SAME task/data/metric, both measured across seeds, confidence intervals
        deciding the verdict (load-bearing / decorative / tie). The honest answer to 'where is
        VSA actually the reason it works'. See holographic_ablate.ablation_table."""
        import holographic.misc.holographic_ablate as _ab
        return _ab.ablation_table(seeds=seeds)

    def roles_by_shift(self, pairs, dim=None):
        """Encode role-filler pairs with ROLES AS POWERS OF ONE SHIFT OPERATOR -- the trick that
        made the in-weights role machine affordable (one permutation instead of one circulant
        per role; the origin design behind the weight installs). Returns the trace; decode with
        holographic_vsaroles.decode_structure. See holographic_vsaroles.encode_structure."""
        import holographic.io_and_interop.holographic_vsaroles as _vr
        return _vr.encode_structure(pairs, dim=dim)


def _as_blob(blob):
    """Wire-tolerant blob coercion: bytes pass through; a base64 str or the service's
    {"__bytes_b64__": ...} sentinel (see holographic_service._jsonable) decode to bytes --
    so a blob that crossed HTTP feeds straight back into any *_decode faculty."""
    import base64
    if isinstance(blob, dict) and "__bytes_b64__" in blob:
        return base64.b64decode(blob["__bytes_b64__"])
    if isinstance(blob, str):
        return base64.b64decode(blob)
    return bytes(blob)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract -- then proves
    one representative faculty end-to-end through a real mind (wiring, not just membership)."""
    n = check_part("holographic.unified.holographic_unified_p15_hdrift", "_UnifiedPart15")
    import tempfile, os
    import numpy as np
    from lecore import UnifiedMind as _UM
    m = _UM(dim=128, seed=0)
    rng = np.random.default_rng(0)
    pts = np.vstack([c + 0.04 * rng.standard_normal((40, 2))
                     for c in ([0.25, 0.25], [0.75, 0.75])])
    model = m.drift_train(pts, dim=1024)
    X = m.drift_generate(model, n=12, seed=3)
    a = m.generation_audit(X, pts, k_modes=2)
    assert a["coverage"] >= 0.5 and a["memorised_frac"] < 0.5, "faculty round-trip audit: %s" % a
    p = os.path.join(tempfile.gettempdir(), "p15_wav_selftest.wav")
    m.write_wav(p, np.sin(np.linspace(0, 2 * np.pi * 440, 8000)), 8000)
    s, r = m.read_wav(p)
    assert r == 8000 and abs(len(s) - 8000) <= 1, "wav round-trip through the mind"
    print("holographic_unified_p15_hdrift selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
