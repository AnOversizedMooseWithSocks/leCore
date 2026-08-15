import json

import numpy as np
import pytest

from tools import benchmark_qwen_runtime as bench


def _row(chunk, tps, peak_gib, losses=(1.0, 2.0, 3.0)):
    values = np.asarray(losses, dtype=np.float64)
    return {
        "status": "ok",
        "chunk_size": chunk,
        "tokens_per_second": tps,
        "peak_rss_bytes": int(peak_gib * bench.GIB),
        "_losses": values.tolist(),
        "losses_sha256_float64": bench.hashlib.sha256(values.tobytes()).hexdigest(),
    }


def test_memory_recommendation_accounts_for_parallel_evaluators_and_headroom():
    # Two 10 GiB evaluator processes, 25% headroom, and 2 GiB for the OS need
    # 27 GiB.  The 32 GiB class is therefore the first safe class.
    report = bench.memory_recommendation(
        [_row(128, 2.0, 10.0)], [32, 64, 128], concurrent_runtimes=2,
        headroom_factor=1.25, system_reserve_gib=2.0)
    assert report["recommended_ram_gib"] == 32
    assert report["required_capacity_bytes"] == 27 * bench.GIB
    assert report["class_decisions"][0]["fits"] is True


def test_full_run_peak_can_override_the_evaluation_projection():
    report = bench.memory_recommendation(
        [_row(128, 2.0, 4.0)], [32, 64, 128], concurrent_runtimes=2,
        headroom_factor=1.25, system_reserve_gib=2.0,
        full_run_peak_mib=40 * 1024)
    assert report["recommended_ram_gib"] == 64
    assert report["planning_peak_bytes"] == 40 * bench.GIB


def test_selected_parity_chunk_drives_sizing_but_worst_trial_is_preserved():
    rows = [_row(128, 8.0, 4.0), _row(256, 7.0, 20.0)]
    report = bench.memory_recommendation(
        rows, [8, 32], concurrent_runtimes=1, headroom_factor=1.0,
        system_reserve_gib=0.0, selected_chunk_size=128)
    assert report["recommended_ram_gib"] == 8
    assert report["measured_single_runtime_peak_bytes"] == 4 * bench.GIB
    assert report["max_observed_trial_peak_bytes"] == 20 * bench.GIB


def test_chunk_recommendation_requires_loss_parity_and_projects_cost():
    rows = [
        _row(64, 4.0, 3.0),
        _row(128, 8.0, 4.0),
        _row(256, 20.0, 6.0, losses=(1.0, 2.0, 3.1)),
    ]
    assert bench.annotate_parity(rows, atol=1e-9) == 64
    report = bench.speed_recommendation(
        rows, parity_atol=1e-9, eval_tokens_per_runtime=4096,
        eval_passes=2, concurrent_runtimes=2, fixed_overhead_minutes=10,
        rates={32: 0.50})
    assert report["recommended_chunk_size"] == 128
    assert report["execution_waves"] == 1
    assert report["estimated_total_seconds"] == pytest.approx(1112.0)
    assert report["cost_estimates"][0]["estimated_compute_usd"] == pytest.approx(
        0.5 * 1112.0 / 3600.0)
    assert all("_losses" not in row for row in rows)


def test_cost_fit_marks_only_classes_allowed_by_memory_evidence():
    speed = {"cost_estimates": [
        {"ram_gib": 32, "estimated_compute_usd": 0.2},
        {"ram_gib": 64, "estimated_compute_usd": 0.4},
    ]}
    memory = {"class_decisions": [
        {"ram_gib": 32, "fits": False}, {"ram_gib": 64, "fits": True},
    ]}
    bench.annotate_cost_fit(speed, memory)
    assert speed["cost_estimates"][0]["fits_memory_projection"] is False
    assert speed["cost_estimates"][1]["fits_memory_projection"] is True
    assert speed["least_estimated_cost_fitting_class_gib"] == 64


def test_parsers_reject_ambiguous_or_nonpositive_inputs():
    assert bench.parse_positive_ints("128,32,128") == [32, 128]
    assert bench.parse_rates("32=0.25,64=0.5") == {32: 0.25, 64: 0.5}
    with pytest.raises(Exception):
        bench.parse_positive_ints("32,zero")
    with pytest.raises(Exception):
        bench.parse_rates("32 dollars")


def test_model_manifest_is_sorted_and_checksummed(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "ignored.txt").write_text("ignore", encoding="utf-8")
    rows = bench.model_manifest(tmp_path)
    assert [row["name"] for row in rows] == [
        "config.json", "model-00001.safetensors"]
    assert all(len(row["sha256"]) == 64 for row in rows)
    # It is valid JSON evidence without leaking the weight contents.
    json.dumps(rows)


def test_token_losses_match_float64_reference_without_promoting_the_slab():
    logits = np.asarray([[1.0, 2.0, -3.0], [0.25, -0.5, 0.75]], np.float32)
    targets = np.asarray([1, 0])
    got = bench._token_losses(logits, targets)
    values = logits.astype(np.float64)
    maximum = values.max(axis=-1, keepdims=True)
    expected = (np.log(np.exp(values - maximum).sum(axis=-1)) +
                maximum.ravel() - values[np.arange(2), targets])
    assert got.dtype == np.float64
    assert np.allclose(got, expected, atol=1e-7, rtol=0)
