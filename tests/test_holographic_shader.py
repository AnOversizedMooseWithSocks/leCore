"""H6 -- N filter passes in one evaluation, plus the two things a GPU structurally cannot do."""
import numpy as np
import pytest

from holographic.rendering.holographic_shader import blur_kernel, filter_k, filter_limit


def _setup(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), blur_kernel((n,))


def _literal(x, k, passes):
    for _ in range(passes):
        x = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(k)))
    return x


def test_integer_passes_match_the_literal_loop_and_cost_is_independent_of_n():
    f, k = _setup()
    for N in (1, 8, 64, 512):
        assert np.max(np.abs(filter_k(f, k, N) - _literal(f, k, N))) < 1e-9
    assert np.isfinite(filter_k(f, k, 1_000_000)).all()      # a million passes, no loop


def test_fractional_passes_compose():
    """Half a blur pass, twice, is exactly one pass. There is no GPU analogue."""
    f, k = _setup()
    half = filter_k(f, k, 0.5)
    assert np.max(np.abs(filter_k(half, k, 0.5) - filter_k(f, k, 1))) < 1e-9
    third = filter_k(f, k, 1.0 / 3.0)
    assert np.max(np.abs(filter_k(filter_k(third, k, 1 / 3), k, 1 / 3) - filter_k(f, k, 1))) < 1e-9


def test_fractional_pass_refuses_a_sign_changing_transfer():
    """A real transfer that changes sign has no canonical fractional power -- raise, don't pick a branch."""
    f, _ = _setup()
    signed = np.zeros(len(f)); signed[1] = 0.5; signed[-1] = 0.5      # transfer cos(w)
    with pytest.raises(ValueError):
        filter_k(f, signed, 0.5)
    sharpen = np.zeros(len(f)); sharpen[0] = 2.0; sharpen[1] = -0.5; sharpen[-1] = -0.5   # 2 - cos(w) > 0
    assert np.isfinite(filter_k(f, sharpen, 0.5)).all()               # unambiguous -> permitted


def test_infinite_passes_is_an_idempotent_projection():
    f, k = _setup()
    lim = filter_limit(f, k)
    assert np.max(np.abs(filter_limit(lim, k) - lim)) < 1e-12         # idempotent -> a projection
    # a blur's only non-decaying mode is DC, so the limit is the field's mean
    assert abs(lim.mean() - f.mean()) < 1e-9 and lim.std() < 1e-6
    # the loop heads there, but slowly: this blur's slowest mode decays as 0.999849^N (~200k passes)
    assert np.max(np.abs(_literal(f, k, 300) - lim)) < np.max(np.abs(f - lim))
    assert np.max(np.abs(filter_k(f, k, 200_000) - lim)) < 1e-9


def test_amplifying_filter_has_no_limit_and_warns_on_overflow():
    f, _ = _setup()
    amp = np.zeros(len(f)); amp[0] = 1.5
    with pytest.raises(ValueError):
        filter_limit(f, amp)
    with pytest.warns(RuntimeWarning):
        filter_k(f, amp, 100_000)


def test_two_dimensional_fields():
    rng = np.random.default_rng(1)
    F = rng.standard_normal((32, 32))
    K = np.zeros((32, 32)); K[0, 0] = 0.5; K[0, 1] = 0.25; K[0, -1] = 0.25
    once = np.real(np.fft.ifftn(np.fft.fftn(F) * np.fft.fftn(K)))
    twice = np.real(np.fft.ifftn(np.fft.fftn(once) * np.fft.fftn(K)))
    assert np.max(np.abs(filter_k(F, K, 2) - twice)) < 1e-9


def test_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    f, k = _setup(128)
    assert np.max(np.abs(m.filter_passes(f, k, 4) - filter_k(f, k, 4))) < 1e-12
    assert abs(m.filter_limit(f, k).mean() - f.mean()) < 1e-9


# ---------------------------------------------------------------------------------------------------------------
# H3 -- the algebra has a Nyquist: the bake's bandwidth must exceed the signal's max angular frequency.
# ---------------------------------------------------------------------------------------------------------------
def _sine(freq, n=240):
    xs = np.linspace(0.0, 1.0, n)
    return xs, np.sin(2 * np.pi * freq * xs)


def _fetch_rms(bake, freq):
    from holographic.rendering.holographic_shader import fetch
    q = np.linspace(0.05, 0.95, 41)
    pred, true = fetch(bake, q), np.sin(2 * np.pi * freq * q)
    scale = np.dot(pred, true) / np.dot(pred, pred)          # the bake carries an arbitrary gain
    return float(np.sqrt(np.mean((scale * pred - true) ** 2)))


def test_bandwidth_probe_recovers_the_max_angular_frequency():
    from holographic.rendering.holographic_shader import bandwidth_probe
    for freq in (2.0, 5.0, 8.0):
        xs, ys = _sine(freq)
        assert abs(bandwidth_probe(xs, ys) - 2 * np.pi * freq) < 0.15 * 2 * np.pi * freq
    xs = np.linspace(0, 1, 64)
    assert bandwidth_probe(xs, np.ones_like(xs)) == 0.0      # a constant has no frequency content


def test_bake_with_data_chosen_bandwidth_is_accurate_at_every_frequency():
    from holographic.rendering.holographic_shader import bake_1d
    for freq in (2.0, 5.0, 12.0):
        xs, ys = _sine(freq)
        b = bake_1d(xs, ys, dim=4096)
        assert b["bandwidth"] >= b["omega_max"]               # the Nyquist condition, enforced
        assert _fetch_rms(b, freq) < 0.06


def test_below_nyquist_the_bake_is_wrong_and_says_so():
    """The kept negative. Below omega_max the fetch is smooth, confident and wrong -- and nothing raises. The only
    defence is the warning, so it is a hard requirement, not a nicety."""
    import warnings as _w
    from holographic.rendering.holographic_shader import bake_1d
    xs, ys = _sine(12.0)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        bad = bake_1d(xs, ys, dim=4096, bandwidth=3.0)
        assert any("Nyquist" in str(w.message) for w in caught)
    good = bake_1d(xs, ys, dim=4096)
    assert _fetch_rms(bad, 12.0) > 4.0 * _fetch_rms(good, 12.0)   # far worse, with no error raised


def test_bake_and_fetch_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    xs, ys = _sine(3.0)
    b = m.bake_field(xs, ys, dim=2048)
    y = m.fetch_field(b, 0.37)                                # a point never sampled
    assert np.isfinite(y)
    assert _fetch_rms(b, 3.0) < 0.08


# ---------------------------------------------------------------------------------------------------------------
# H1 -- the compiler: a filter graph collapses to one transfer before any data is touched.
# ---------------------------------------------------------------------------------------------------------------
def test_three_stage_graph_compiles_to_one_transfer_exactly():
    from holographic.rendering.holographic_shader import Pipeline
    f, kb = _setup(512)
    n = len(f)
    kw = np.zeros(n); kw[0] = 0.34; kw[1] = 0.33; kw[-1] = 0.33
    w = 2.0 * np.pi * np.fft.fftfreq(n)

    staged = filter_k(f, kb, 8)
    staged = np.real(np.fft.ifft(np.fft.fft(staged) * np.exp(-1j * w * 3)))
    staged = 1.6 * staged - 0.6 * np.real(np.fft.ifft(np.fft.fft(staged) * np.fft.fft(kw)))

    piped = Pipeline((n,)).blur(kb, 8).translate(3).unsharp(kw, 0.6).apply(f)
    assert np.max(np.abs(piped - staged)) < 1e-9


def test_compose_the_operators_not_the_images():
    """The compiler's whole argument. A half-sample shift creates a genuinely IMAGINARY Nyquist-bin component.
    Composed inside the pipeline the two halves are exact; materialising the intermediate takes a real part and
    silently throws that component away (measured ~9e-2 of signal)."""
    from holographic.rendering.holographic_shader import Pipeline
    f, _ = _setup(256)
    n = len(f)
    whole = Pipeline((n,)).translate(1.0).apply(f)
    composed = Pipeline((n,)).translate(0.5).translate(0.5).apply(f)
    materialised = Pipeline((n,)).translate(0.5).apply(Pipeline((n,)).translate(0.5).apply(f))
    assert np.max(np.abs(composed - whole)) < 1e-9            # exact in the transfer domain
    assert np.max(np.abs(materialised - whole)) > 1e-3        # lossy through an intermediate image


def test_pipeline_gain_and_shape_guard():
    from holographic.rendering.holographic_shader import Pipeline
    f, kb = _setup(128)
    n = len(f)
    assert np.allclose(Pipeline((n,)).gain(2.0).apply(f), 2.0 * f)
    with pytest.raises(ValueError):
        Pipeline((n,)).apply(np.zeros(n + 1))
    with pytest.raises(ValueError):                            # fractional pass, sign-changing transfer
        signed = np.zeros(n); signed[1] = 0.5; signed[-1] = 0.5
        Pipeline((n,)).blur(signed, 0.5)


def test_pipeline_two_dimensional_and_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(2)
    F = rng.standard_normal((32, 32))
    K = np.zeros((32, 32)); K[0, 0] = 0.5; K[0, 1] = 0.25; K[0, -1] = 0.25
    piped = m.shader_pipeline(F.shape).blur(K, 3).apply(F)
    assert np.max(np.abs(piped - filter_k(F, K, 3))) < 1e-9


# ======================================================================================================
# H2 -- the superposed gather. N weighted lookups in one dot product.
# ======================================================================================================
def _gfun(t):
    return np.sin(2 * np.pi * 2.0 * t) + 0.4 * np.cos(2 * np.pi * 3.0 * t)


def _gbake(n=400, dim=4096):
    from holographic.rendering.holographic_shader import bake_1d
    xs = np.linspace(0.0, 1.0, n)
    return bake_1d(xs, _gfun(xs), dim=dim, seed=0)


def test_gather_is_exact_against_the_staged_fetches():
    """The load-bearing claim: a gather approximates NOTHING. <F, sum w_j Z(u_j)> == sum w_j <F, Z(u_j)>."""
    from holographic.rendering.holographic_shader import fetch, gather, gather_rule
    b = _gbake()
    rng = np.random.default_rng(7)
    for N in (4, 32, 128, 512):
        u, w = rng.uniform(0.05, 0.95, N), rng.standard_normal(N)
        assert abs(gather(b, gather_rule(b, u, w)) - float(np.sum(w * fetch(b, u)))) < 1e-9


def test_gather_has_no_capacity_wall_error_falls_with_more_taps():
    """The backlog predicted crosstalk ~ sqrt(N/D). Measurement says no: a gather never unbinds, so its error
    AVERAGES DOWN. This test pins the DIRECTION, which is where a crosstalk law would announce itself."""
    from holographic.rendering.holographic_shader import gather, gather_rule
    b = _gbake()

    def err(N, seed):
        r = np.random.default_rng(seed)
        u = r.uniform(0.1, 0.9, N)
        w = r.random(N)
        w /= w.sum()                                        # an interpolation rule: weights sum to 1
        return abs(gather(b, gather_rule(b, u, w), normalize=True) - float(np.sum(w * _gfun(u))))

    errs = [np.mean([err(N, 300 + s) for s in range(16)]) for N in (2, 8, 32, 128, 512)]
    assert errs[-1] < errs[0] / 3.0, errs                   # measured 0.053 -> 0.008
    assert all(errs[i + 1] <= errs[i] * 1.2 for i in range(len(errs) - 1)), errs   # monotone-ish, never sqrt(N)


def test_translate_rule_is_literally_a_bind_and_costs_nothing_per_tap():
    from holographic.agents_and_reasoning.holographic_ai import bind, cosine
    from holographic.rendering.holographic_shader import gather, gather_rule, translate_rule
    b = _gbake()
    rng = np.random.default_rng(11)
    u, w = rng.uniform(0.1, 0.6, 64), rng.standard_normal(64)
    Q = gather_rule(b, u, w)
    dx = 0.05                                               # in range, so encode(dx) does not warn
    assert cosine(translate_rule(b, Q, dx)["rule"], bind(Q["rule"], b["encoder"].encode(dx))) > 1.0 - 1e-9
    rebuilt = gather_rule(b, u + dx, w)                     # every tap re-encoded at the shifted position
    assert cosine(translate_rule(b, Q, dx)["rule"], rebuilt["rule"]) > 1.0 - 1e-9
    assert abs(gather(b, translate_rule(b, Q, dx)) - gather(b, rebuilt)) < 1e-9
    back = translate_rule(b, translate_rule(b, Q, -0.2), 0.2)
    assert cosine(back["rule"], Q["rule"]) > 1.0 - 1e-9                                  # invertible


def test_raw_fetch_carries_the_sample_count_as_a_gain_and_normalize_removes_it():
    """KEPT LOUD: the raw fetch is a kernel SUM. Its scale is how densely you sampled, not a property of f."""
    from holographic.rendering.holographic_shader import fetch
    q = np.linspace(0.1, 0.9, 41)
    coarse, fine = _gbake(n=100), _gbake(n=800)
    ratio = float(np.dot(fetch(fine, q), fetch(coarse, q)) / np.dot(fetch(coarse, q), fetch(coarse, q)))
    assert 7.0 < ratio < 9.0, ratio                          # 8x the samples -> 8x the raw fetch
    for bk in (coarse, fine):                                # normalized: lands on f, with no fitted constant
        assert np.sqrt(np.mean((fetch(bk, q, normalize=True) - _gfun(q)) ** 2)) / np.std(_gfun(q)) < 0.10


def test_normalized_fetch_rescues_clumped_samples_where_the_raw_fetch_is_useless():
    from holographic.rendering.holographic_shader import bake_1d, bandwidth_probe, fetch
    grid = np.linspace(0, 1, 400)
    B = 1.5 * bandwidth_probe(grid, _gfun(grid))             # the probe is an FFT: take B from a UNIFORM proxy
    rng = np.random.default_rng(3)
    xs = np.sort(np.concatenate([rng.uniform(0, 0.3, 300), rng.uniform(0.3, 1.0, 100)]))   # 3:1 clumped
    b = bake_1d(xs, _gfun(xs), dim=4096, seed=0, bandwidth=B)
    q = np.linspace(0.05, 0.95, 61)
    truth = _gfun(q)
    raw = fetch(b, q)
    C = float(np.dot(raw, truth) / np.dot(truth, truth))     # the kindest possible rescale: fitted to the truth
    rms = lambda a: float(np.sqrt(np.mean((a - truth) ** 2)) / np.std(truth))
    assert rms(raw / C) > 1.0                                # worse than predicting the mean, even so
    assert rms(fetch(b, q, normalize=True)) < 0.4            # ...and fine once divided by the density


def test_gather_rule_rejects_mismatched_weights_and_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    from holographic.rendering.holographic_shader import gather_rule
    b = _gbake(dim=1024)
    with pytest.raises(ValueError):
        gather_rule(b, [0.1, 0.2, 0.3], [1.0, 1.0])
    m = UnifiedMind(dim=64, seed=0)
    xs = np.linspace(0.0, 1.0, 200)
    mb = m.bake_field(xs, _gfun(xs), dim=2048)
    Q = m.gather_rule(mb, [0.2, 0.5, 0.8], [0.25, 0.5, 0.25])
    staged = float(np.sum(np.array([0.25, 0.5, 0.25]) * m.fetch_field(mb, np.array([0.2, 0.5, 0.8]))))
    assert abs(m.gather_field(mb, Q) - staged) < 1e-9
    assert abs(m.gather_field(mb, m.translate_rule(mb, Q, 0.0)) - staged) < 1e-9


def test_scattered_samples_warn_about_the_probe_not_about_nyquist():
    """The probe is an FFT. On scattered xs it over-reports w_max (100.8 against a true 18.8) -- so bake_1d must
    not raise a Nyquist alarm off a number it cannot trust, and must say so when asked to choose the bandwidth."""
    import warnings as _w
    from holographic.rendering.holographic_shader import bake_1d, bandwidth_probe
    grid = np.linspace(0, 1, 400)
    B = 1.5 * bandwidth_probe(grid, _gfun(grid))
    rng = np.random.default_rng(3)
    xs = np.sort(np.concatenate([rng.uniform(0, 0.3, 300), rng.uniform(0.3, 1.0, 100)]))
    with _w.catch_warnings(record=True) as caught:           # explicit bandwidth on scattered xs: NO false alarm
        _w.simplefilter("always")
        bake_1d(xs, _gfun(xs), dim=1024, bandwidth=B)
        assert not any("Nyquist" in str(c.message) for c in caught)
    with _w.catch_warnings(record=True) as caught:           # asking it to choose: it says it cannot
        _w.simplefilter("always")
        bake_1d(xs, _gfun(xs), dim=1024)
        assert any("not uniformly spaced" in str(c.message) for c in caught)


def test_a_rule_from_another_encoder_refuses_rather_than_answer():
    """MEASURED: applied to a bake with a different bandwidth, a rule returned 1.816 where the truth was -0.818 --
    finite, confident, wrong, and silent. Signature-checked now; share the encoder and it transfers exactly."""
    from holographic.rendering.holographic_shader import bake_1d, fetch, gather, gather_rule
    b = _gbake()
    rng = np.random.default_rng(5)
    u, w = rng.uniform(0.1, 0.9, 8), rng.standard_normal(8)
    Q = gather_rule(b, u, w)

    xs = np.linspace(0.0, 1.0, 400)
    other = bake_1d(xs, np.cos(2 * np.pi * xs), dim=4096, seed=0)              # its own bandwidth -> its own encoder
    with pytest.raises(ValueError):
        gather(other, Q)

    shared = bake_1d(xs, np.cos(2 * np.pi * xs), dim=4096, seed=0, bandwidth=b["bandwidth"])
    assert abs(gather(shared, Q) - float(np.sum(w * fetch(shared, u)))) < 1e-9
    assert abs(gather(shared, Q["rule"]) - gather(shared, Q)) < 1e-12          # bare vector: guard opted out


def test_gather_samples_is_the_stateless_form_and_survives_a_real_http_invoke():
    """/tools advertises bake_field + gather_rule, but they hand back LIVE objects that serialise to dead dicts.
    gather_samples is the one-shot twin: plain JSON in, a plain number out. Proven over a real socket, because
    'it works in-process' and 'an agent can call it' are different claims."""
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer

    import holographic_service as svc_mod
    from holographic.rendering.holographic_shader import gather_samples

    xs = np.linspace(0.0, 1.0, 400)
    ys = _gfun(xs)
    taps = [0.3, 0.4, 0.5, 0.6, 0.7]
    w = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
    truth = float(np.sum(np.array(w) * _gfun(np.array(taps))))

    assert abs(gather_samples(xs, ys, taps, w, dim=4096) - truth) < 0.05        # in process

    svc = svc_mod.Service()
    httpd = HTTPServer(("127.0.0.1", 0), svc_mod.make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    try:
        tools = json.loads(urllib.request.urlopen(base + "/tools", timeout=30).read())
        assert "gather_samples" in {t["name"] for t in tools["tools"]}
        body = json.dumps({"name": "gather_samples",
                           "args": {"xs": list(xs), "ys": list(ys), "points": taps, "weights": w, "dim": 4096}})
        req = urllib.request.Request(base + "/invoke", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        got = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert got["ok"] and abs(float(got["result"]) - truth) < 0.05, got
    finally:
        httpd.shutdown()
        httpd.server_close()


# ======================================================================================================
# H7 -- M variants combine exactly; the superposed variant BANK is a kept negative.
# ======================================================================================================
def _blur_stack(n, sigmas):
    from holographic.rendering.holographic_shader import Pipeline, gauss_kernel
    return [Pipeline((n,)).blur(gauss_kernel(n, s)) for s in sigmas]


def test_combine_folds_m_variants_into_one_transfer_exactly():
    from holographic.rendering.holographic_shader import combine
    rng = np.random.default_rng(0)
    n = 512
    f = rng.standard_normal(n)
    pipes = _blur_stack(n, (2.0, 6.0, 14.0, 30.0))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    staged = sum(wi * p.apply(f) for wi, p in zip(w, pipes))
    assert np.max(np.abs(combine(pipes, w).apply(f) - staged)) < 1e-12
    assert np.max(np.abs(combine(pipes).apply(f) - sum(p.apply(f) for p in pipes))) < 1e-12   # weights default to 1
    # ...and the win is that the combined cost does not depend on M: one transfer, one multiply, one inverse FFT.
    assert combine(pipes, w).transfer.shape == (n,)


def test_combine_keeps_chaining_and_rejects_mismatches():
    from holographic.rendering.holographic_shader import blur_kernel, combine, Pipeline
    rng = np.random.default_rng(1)
    n = 256
    f = rng.standard_normal(n)
    pipes = _blur_stack(n, (2.0, 8.0))
    blended = combine(pipes, [0.5, 0.5])
    chained = combine(pipes, [0.5, 0.5]).blur(blur_kernel((n,)), 2)      # still a Pipeline
    direct = Pipeline((n,)).blur(blur_kernel((n,)), 2).apply(blended.apply(f))
    assert np.max(np.abs(chained.apply(f) - direct)) < 1e-9
    with pytest.raises(ValueError):
        combine(pipes, [1.0])                                            # weight/pipeline mismatch
    with pytest.raises(ValueError):
        combine([])                                                      # nothing to combine
    with pytest.raises(ValueError):
        combine([Pipeline((n,)), Pipeline((n + 1,))])                    # shapes must agree


def test_superposed_variant_bank_is_a_kept_negative_fidelity_is_one_over_sqrt_m():
    """H7's other half, pinned so it is never re-derived. Bundling M keyed variants and unbinding one back out
    recovers it at ~1/sqrt(M) -- NOT at 1 - sqrt(M/D), which is the cosine with a WRONG item, a different quantity.
    Measured on UNCORRELATED transfers, i.e. the kindest possible case for the bank."""
    from holographic.rendering.holographic_shader import _phasor_key
    D = 2048
    rng = np.random.default_rng(1)
    Fh = np.fft.fft(rng.standard_normal(D))
    for M in (2, 8, 32):
        Hs = [np.fft.fft(rng.standard_normal(D)) for _ in range(M)]
        Ks = [_phasor_key(D, 100 + j) for j in range(M)]
        bank = sum(K * H for K, H in zip(Ks, Hs))
        fids = []
        for j in range(M):
            rec = np.real(np.fft.ifft(np.conj(Ks[j]) * (bank * Fh)))
            tru = np.real(np.fft.ifft(Hs[j] * Fh))
            fids.append(float(np.dot(rec, tru) / (np.linalg.norm(rec) * np.linalg.norm(tru))))
        assert abs(np.mean(fids) - 1.0 / np.sqrt(M)) < 0.05, (M, np.mean(fids))
        assert np.mean(fids) < 0.98                                       # never usable, not even at M=2


def test_real_shader_variants_are_correlated_which_is_why_cleanup_cannot_rescue_the_bank():
    """The second reason the bank fails: M blurs of the SAME field are filtered copies of one signal, not M
    independent items. Cleanup -- the discrete decision that normally resets crosstalk -- needs near-orthogonality."""
    rng = np.random.default_rng(2)
    n = 1024
    f = rng.standard_normal(n)
    outs = [p.apply(f) for p in _blur_stack(n, (2.0, 8.0))]
    cos2 = abs(float(np.dot(outs[0], outs[1]) / (np.linalg.norm(outs[0]) * np.linalg.norm(outs[1]))))
    assert cos2 > 0.3, cos2                                               # measured ~0.49; nowhere near orthogonal
    # ...whereas the thing that DOES work needs no orthogonality at all, because it never unbinds:
    from holographic.rendering.holographic_shader import combine
    pipes = _blur_stack(n, (2.0, 8.0))
    assert np.max(np.abs(combine(pipes, [0.7, 0.3]).apply(f) - (0.7 * outs[0] + 0.3 * outs[1]))) < 1e-12


def test_shader_combine_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(3)
    n = 256
    f = rng.standard_normal(n)
    pipes = _blur_stack(n, (2.0, 6.0, 14.0))
    w = [0.5, 0.3, 0.2]
    staged = sum(wi * p.apply(f) for wi, p in zip(w, pipes))
    assert np.max(np.abs(m.shader_combine(pipes, w).apply(f) - staged)) < 1e-12
    assert any("variant" in c.name.lower() or "blend" in c.name.lower()
               for c in m.find_capability("combine shader variants"))


# ======================================================================================================
# H4 -- detrend before you bake. It is the wrap, not the singularity.
# These shipped with a self-test and NO pytest coverage; this file is that coverage.
# ======================================================================================================
_DX = np.linspace(0.0, 1.0, 400)
_DQ = np.linspace(0.002, 0.998, 200)


def _arel(got, truth):
    return float(np.sqrt(np.mean((got - truth) ** 2)) / np.std(truth))


def test_the_probe_is_fooled_by_the_wrap_not_by_the_singularity():
    """A straight line has no high frequencies at all, yet probes near sqrt's bandwidth -- because an FFT treats its
    samples as periodic, and a mismatched pair of endpoints IS a jump discontinuity."""
    from holographic.rendering.holographic_shader import bandwidth_probe
    w_line = bandwidth_probe(_DX, _DX)
    w_sqrt = bandwidth_probe(_DX, np.sqrt(_DX))
    w_sine = bandwidth_probe(_DX, np.sin(4 * np.pi * _DX))          # genuinely 2 cycles: w_max = 4*pi = 12.57
    assert w_line > 500.0 and w_sqrt > 500.0, (w_line, w_sqrt)
    assert w_sine < 20.0, w_sine
    assert w_line > 20 * w_sine                                     # the line "looks" 48x more oscillatory than a sine
    # remove the endpoint line and the phantom bandwidth vanishes entirely
    line = _DX[0] + (_DX[-1] - _DX[0]) * (_DX - _DX[0]) / (_DX[-1] - _DX[0])
    assert bandwidth_probe(_DX, _DX - line) < 1e-9


def test_detrending_wins_at_every_seed_and_absolute_bars_would_not():
    """The claim is a CONTRAST, not a number. The plain bake's error ranges 0.27-4.49 across encoder seeds, so any
    single-seed threshold is a lottery ticket -- which is exactly how a wrong table shipped here once."""
    import warnings as _w
    from holographic.rendering.holographic_shader import bake_1d, fetch
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        for f in (np.sqrt, np.cbrt, lambda t: 1.0 / (t + 0.05)):
            ratios = []
            for seed in range(6):
                plain = _arel(fetch(bake_1d(_DX, f(_DX), dim=4096, seed=seed), _DQ, normalize=True), f(_DQ))
                detr = _arel(fetch(bake_1d(_DX, f(_DX), dim=4096, seed=seed, detrend=True), _DQ,
                                   normalize=True), f(_DQ))
                assert detr < plain, (seed, plain, detr)             # never worse, at any seed
                ratios.append(plain / max(detr, 1e-12))
            assert np.median(ratios) > 4.0, ratios                   # typically 5-17x


def test_detrending_is_exact_on_a_line_and_free_on_a_periodic_function():
    import warnings as _w
    from holographic.rendering.holographic_shader import bake_1d, fetch
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        b = bake_1d(_DX, _DX, dim=4096, seed=0, detrend=True)        # residual is identically zero
        assert _arel(fetch(b, _DQ, normalize=True), _DQ) < 1e-9
        sn = lambda t: np.sin(2 * np.pi * 2 * t)                     # already periodic: no trend to remove
        e_p = _arel(fetch(bake_1d(_DX, sn(_DX), dim=4096, seed=0), _DQ, normalize=True), sn(_DQ))
        e_d = _arel(fetch(bake_1d(_DX, sn(_DX), dim=4096, seed=0, detrend=True), _DQ, normalize=True), sn(_DQ))
        assert abs(e_p - e_d) < 1e-9


def test_only_the_detrended_bake_gets_better_with_dimension():
    """You cannot buy your way out of a bad bandwidth with dimension. Measured on sqrt: plain 0.199 -> 0.069 across
    16x the dimension, detrended 0.026 -> 0.006 -- and the detrended bake is 10x better throughout."""
    import warnings as _w
    from holographic.rendering.holographic_shader import bake_1d, fetch
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        mean = lambda dim, det: np.mean([_arel(fetch(bake_1d(_DX, np.sqrt(_DX), dim=dim, seed=s, detrend=det),
                                                     _DQ, normalize=True), np.sqrt(_DQ)) for s in range(4)])
        assert mean(16384, True) < mean(1024, True) / 2.0            # detrended: dimension pays
        assert mean(16384, True) < mean(16384, False) / 3.0          # ...and it is still far ahead


def test_a_detrended_bake_refuses_a_raw_fetch_and_a_gather():
    """The trend is an absolute offset. A raw kernel SUM has no absolute scale, and a compiled gather rule is one
    vector that no longer knows its own sample points -- so both must raise rather than silently drop the line."""
    import warnings as _w
    from holographic.rendering.holographic_shader import bake_1d, fetch, gather, gather_rule
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        b = bake_1d(_DX, np.sqrt(_DX), dim=1024, seed=0, detrend=True)
    with pytest.raises(ValueError):
        fetch(b, 0.5)                                                # raw fetch: refuses
    with pytest.raises(ValueError):
        gather(b, gather_rule(b, [0.3, 0.6]))                        # gather: refuses
    assert abs(fetch(b, 0.5, normalize=True) - np.sqrt(0.5)) < 0.05  # normalized: fine


# ======================================================================================================
# H5 -- the n-D texture unit. Bandwidth is a bias-variance dial; dim is the variance budget.
# ======================================================================================================
def _gg(P):
    P = np.asarray(P, float)
    return np.sin(2 * np.pi * P[..., 0]) * np.cos(2 * np.pi * P[..., 1])


def _nd_setup(n=40):
    ax = np.linspace(0.0, 1.0, n)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    return ax, _gg(P)


def _sf(got, truth):
    """Scale-free RMS: 1.0 means the readout carries no information about the truth."""
    c = float(np.dot(got, truth) / np.dot(got, got)) if np.dot(got, got) > 0 else 0.0
    return float(np.sqrt(np.mean((c * got - truth) ** 2)) / np.std(truth))


def test_the_nd_library_default_bandwidth_carries_no_information():
    """The n-D encoder's class default of 3.0 is not a sensible prior -- it is a fixed number pretending to be one.
    Against a 2-cycle sine (w_max = 4*pi = 12.6, i.e. four times the default) it measures ~1.0 scale-free RMS:
    literally no information, silently. Against a 1-cycle sine (w_max = 6.3) it merely mangles it (~0.58). The
    default is not wrong by a little; it is wrong by whatever your data happens to be. Hence bake_nd probes."""
    from holographic.rendering.holographic_shader import bake_nd, fetch_nd
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    ax = np.linspace(0.0, 1.0, 40)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    Q = np.random.default_rng(0).uniform(0.15, 0.85, (120, 2))

    def run(cycles):
        g = lambda A: np.sin(2 * np.pi * cycles * A[..., 0]) * np.cos(2 * np.pi * cycles * A[..., 1])
        V, T = g(P), g(Q)
        bad = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=0)
        Fb, Db = bad.bundle_normalized(P.reshape(-1, 2), V.reshape(-1))
        default = _sf(np.array([bad.query_normalized(Fb, Db, q) for q in Q]), T)
        probed = _sf(fetch_nd(bake_nd([ax, ax], V, dim=8192, seed=0), Q), T)
        return default, probed

    d2, p2 = run(2)
    assert d2 > 0.9, d2                                     # 2 cycles: the default carries nothing at all
    assert p2 < 0.30, p2                                    # ...probed, it carries the shape
    d1, p1 = run(1)
    assert 0.4 < d1 < 0.8, d1                               # 1 cycle: badly attenuated, but not empty
    assert p1 < d1, (p1, d1)                                # probing wins at both frequencies


def test_nd_bandwidth_is_a_bias_variance_dial_and_dim_is_the_variance_budget():
    """At the default margin the error is a BIAS floor: 16x the dimension buys nothing (0.1179 -> 0.1191). Raise the
    margin and the bias falls but crosstalk rises -- which is what dim pays for. Pinned because a previous docstring
    claimed 'error falls with D' at the default margin, and it does not."""
    from holographic.rendering.holographic_shader import bake_nd, fetch_nd
    ax, V = _nd_setup()
    Q = np.random.default_rng(0).uniform(0.05, 0.95, (200, 2))
    T = _gg(Q)
    err = lambda dim, margin: _sf(fetch_nd(bake_nd([ax, ax], V, dim=dim, seed=0, margin=margin), Q), T)
    lo_small, lo_big = err(4096, 1.5), err(65536, 1.5)
    assert abs(lo_big - lo_small) < 0.25 * lo_small, (lo_small, lo_big)     # a bias floor: D changes nothing
    hi_small, hi_big = err(4096, 4.0), err(65536, 4.0)
    assert hi_small > lo_small                                              # wide kernel, small D: WORSE
    assert hi_big < lo_big / 2.0                                            # wide kernel, big D: much better


def test_the_causal_variable_is_the_bandwidth_not_the_margin():
    """Two tables in this codebase once disagreed about whether dimension helps the n-D bake. Both were right about
    their own signal. `margin` is a RATIO -- B = margin * w_max -- so the same margin is a different kernel on
    different data. Hold B fixed and the confound disappears: two different signals at the same B behave the same,
    and one signal at two different B's does not."""
    from holographic.rendering.holographic_shader import fetch_nd
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    ax = np.linspace(0.0, 1.0, 40)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    Q = np.random.default_rng(0).uniform(0.05, 0.95, (200, 2))

    def err(cycles, B, D):
        g = lambda A: np.sin(2 * np.pi * cycles * A[..., 0]) * np.cos(2 * np.pi * cycles * A[..., 1])
        enc = VectorFunctionEncoder(2, dim=D, bounds=[(0, 1), (0, 1)], seed=0, bandwidth=[B, B])
        F, Dn = enc.bundle_normalized(P.reshape(-1, 2), g(P).reshape(-1))
        return _sf(fetch_nd({"encoder": enc, "field": F, "density": Dn}, Q), g(Q))

    # B = 9.4 on a 1-cycle sine: BIAS-limited. Sixteen times the dimension changes nothing.
    lo_small, lo_big = err(1, 9.4, 2048), err(1, 9.4, 32768)
    assert abs(lo_big - lo_small) < 0.30 * lo_small, (lo_small, lo_big)

    # B = 18.8: VARIANCE-limited -- and it pays for BOTH signals, which is the controlled comparison.
    for cycles in (1, 2):
        small, big = err(cycles, 18.8, 2048), err(cycles, 18.8, 32768)
        assert big < small / 1.8, (cycles, small, big)

    # ...so the diagnostic is: double D. If the error drops you are variance-limited; if not, raise the bandwidth.


def test_nd_has_no_capacity_budget_on_the_number_of_bundled_points():
    """A bundled function is only ever summed, never unbound -- so it sits where the H2 gather sits. Held at a fixed
    bandwidth (so the probe cannot confound it), the error is flat as the grid goes 20x20 -> 60x60."""
    from holographic.rendering.holographic_shader import fetch_nd
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    Q = np.random.default_rng(0).uniform(0.05, 0.95, (200, 2))
    T = _gg(Q)
    errs = []
    for n in (20, 40, 60):
        ax = np.linspace(0.0, 1.0, n)
        P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
        enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 1), (0, 1)], seed=0,
                                    bandwidth=[1.5 * 2 * np.pi] * 2)
        F, D = enc.bundle_normalized(P.reshape(-1, 2), _gg(P).reshape(-1))
        errs.append(_sf(fetch_nd({"encoder": enc, "field": F, "density": D}, Q), T))
    assert max(errs) < 2.0 * min(errs), errs                                # flat: 9x the points, no blow-up
    assert max(errs) < 0.20, errs


def test_nd_is_a_shape_estimator_at_the_default_margin_through_the_mind():
    """KEPT NEGATIVE, pinned: the default readout is attenuated (amplitude gain ~0.66). Read shape, not amplitude."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    ax, V = _nd_setup()
    Q = np.random.default_rng(0).uniform(0.05, 0.95, (200, 2))
    T = _gg(Q)
    got = m.fetch_field_nd(m.bake_field_nd([ax, ax], V, dim=8192), Q)
    gain = float(np.dot(got, T) / np.dot(T, T))
    assert 0.4 < gain < 0.9, gain                                            # attenuated, not calibrated
    assert _sf(got, T) < 0.25                                                # ...but the SHAPE is right
    assert any("n-d" in c.name.lower() or "N-D" in c.name for c in m.find_capability("bake a 2d function"))

