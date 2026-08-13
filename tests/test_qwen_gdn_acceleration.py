"""Parity and fallback gates for the optional native Qwen GDN recurrence."""

import numpy as np

from holographic.io_and_interop.holographic_gdnaccel import (
    GDNRecurrence, gdn_recurrence_numpy)


def _inputs(tokens=37, key_dim=16, value_dim=16, seed=7):
    """Qwen3.5 mini-fixture head counts/dimensions (the real model divided by 8)."""
    rng = np.random.default_rng(seed)
    heads = 16
    q = rng.standard_normal((tokens, heads, key_dim), dtype=np.float32) * 0.1
    k = rng.standard_normal(q.shape, dtype=np.float32) * 0.1
    v = rng.standard_normal((tokens, heads, value_dim), dtype=np.float32) * 0.1
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((tokens, heads), dtype=np.float32)))
    g = -np.abs(rng.standard_normal((tokens, heads), dtype=np.float32)) * 0.03
    return q, k, v, beta.astype(np.float32), g.astype(np.float32)


def test_numpy_backend_is_the_unchanged_default():
    arrays = _inputs()
    expected = gdn_recurrence_numpy(*arrays)
    dispatch = GDNRecurrence()
    got = dispatch(*arrays)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    report = dispatch.report()
    assert report["active"] == "numpy"
    assert report["scope"] == "full_sequence_gdn_recurrence"
    assert report["numpy_calls"] == 1


def test_native_scan_parity_including_resumed_state():
    arrays = _inputs(tokens=53)
    first_ref = gdn_recurrence_numpy(*arrays)
    dispatch = GDNRecurrence("c")
    first = dispatch(*arrays)
    if dispatch.report()["refused"]:
        # No system compiler is a supported NumPy-only installation, and the
        # fallback must still be exactly the reference result.
        assert np.array_equal(first[0], first_ref[0])
        assert np.array_equal(first[1], first_ref[1])
        return
    assert np.allclose(first[0], first_ref[0], rtol=5e-5, atol=5e-6)
    assert np.allclose(first[1], first_ref[1], rtol=5e-5, atol=5e-6)

    tail = _inputs(tokens=19, seed=11)
    resumed_ref = gdn_recurrence_numpy(*tail, initial=first_ref[1])
    resumed = dispatch(*tail, initial=first[1])
    assert np.allclose(resumed[0], resumed_ref[0], rtol=5e-5, atol=5e-6)
    assert np.allclose(resumed[1], resumed_ref[1], rtol=5e-5, atol=5e-6)
    report = dispatch.report()
    assert report["active"] == "c"
    assert {row["resumed"] for row in report["validated_regimes"]} == {False, True}
    assert report["native_calls"] == 2
    assert report["native_tokens"] == 72
    assert report["native_libraries"]
    provenance = report["native_libraries"][0]
    assert provenance["flags"] == ["-O2", "-ffp-contract=off"]
    assert len(provenance["library_sha256"]) == 64
    assert provenance["compiler"]["host_machine"]


def test_bad_native_result_is_refused_and_falls_back(monkeypatch):
    arrays = _inputs(tokens=5)
    dispatch = GDNRecurrence("c")

    class Wrong:
        def __call__(self, q, k, v, beta, g, initial):
            return (np.full((q.shape[0], q.shape[1], v.shape[2]), 99, q.dtype),
                    np.full(initial.shape, 99, q.dtype))

    monkeypatch.setattr(dispatch, "_native", lambda dtype: Wrong())
    expected = gdn_recurrence_numpy(*arrays)
    got = dispatch(*arrays)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    assert "parity gate failed" in dispatch.report()["refused"]


def test_mini_qwen_runtime_and_resumed_logits_match(tmp_path):
    """Exercise the substitution inside the structure-faithful Qwen fixture."""
    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from tools.build_mini_qwen import build

    model_dir = tmp_path / "mini-qwen"
    build(model_dir, layers=4, vocab=512)
    reference, config = load_runtime(model_dir)
    native_config = dict(config, gdn_recurrence_backend="c")
    native = GDNRuntime(reference.w, native_config)

    prefix = list(range(32, 52))
    ref_logits, ref_state = reference.forward(prefix, collect_state=True)
    got_logits, got_state = native.forward(prefix, collect_state=True)
    assert np.allclose(got_logits, ref_logits, rtol=1e-4, atol=1e-5)

    tail = [52, 53, 54, 55, 56, 57]
    ref_tail, ref_resumed = reference.forward(
        tail, collect_state=True, resume=ref_state)
    got_tail, got_resumed = native.forward(
        tail, collect_state=True, resume=got_state)
    assert np.allclose(got_tail, ref_tail, rtol=1e-4, atol=1e-5)
    for layer in ref_resumed.gdn:
        assert np.allclose(got_resumed.gdn[layer]["S"],
                           ref_resumed.gdn[layer]["S"], rtol=5e-5, atol=5e-6)
        assert np.allclose(got_resumed.gdn[layer]["conv"],
                           ref_resumed.gdn[layer]["conv"], rtol=1e-4, atol=1e-5)
    report = native.acceleration_report()["full_sequence_gdn_recurrence"]
    if report["refused"] is None:
        assert report["active"] == "c"
        assert {row["resumed"] for row in report["validated_regimes"]} == {False, True}
    else:
        assert report["active"] == "numpy"
