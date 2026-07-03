import argparse
import json

from holographic_metrics import _missing_optional_dependency, build_report, collect_c_mode_tests, collect_path_d


def metrics_smoke_pass():
    pass


def test_path_d_cache_metrics_are_extracted(tmp_path):
    cache = tmp_path / "path_d-cache"
    cache.mkdir()
    (cache / "_tree_cache.json").write_text(
        json.dumps(
            {
                "K": 2401,
                "exhaustive": 0.882,
                "results": [
                    {"depth": 1, "top1_b1": 0.882, "top1_b5": 0.882, "rec_b5": 1.0, "comp_b1": 2401},
                    {"depth": 4, "top1_b1": 0.881, "top1_b5": 0.882, "rec_b5": 0.999, "comp_b1": 28},
                ],
            }
        ),
        encoding="utf-8",
    )
    (cache / "_fwd_cache.json").write_text(
        json.dumps(
            {
                "D": 1024,
                "Cs": [8, 16, 32, 64],
                "Ks": [1, 8],
                "res": {
                    "1": {"fid": [0.96, 0.91, 0.82, 0.7]},
                    "8": {"fid": [0.99, 0.98, 0.94, 0.91]},
                },
            }
        ),
        encoding="utf-8",
    )
    (cache / "_factor_cache.json").write_text(
        json.dumps(
            {
                "D": 1024,
                "V": 8,
                "rows": [
                    {"F": 2, "space": 64, "dense": 1.0, "sbc": 1.0},
                    {"F": 3, "space": 512, "dense": 0.95, "sbc": 1.0},
                    {"F": 4, "space": 4096, "dense": 0.5, "sbc": 0.95},
                ],
            }
        ),
        encoding="utf-8",
    )

    section = collect_path_d(tmp_path)
    metrics = {row["name"]: row for row in section["metrics"]}

    assert metrics["path_d.pivot_tree.depth4_beam5_recall"]["value"] == 0.999
    assert metrics["path_d.distributed_forward.K8_capacity_gain_over_single"]["value"] == 4.0
    assert metrics["path_d.factor_wall.sbc_extra_factors"]["value"] == 1


def test_fast_metrics_report_writes_json_and_markdown(tmp_path):
    args = argparse.Namespace(
        output_dir=tmp_path,
        full_ablations=False,
        include_stress=False,
        run_path_d=None,
        path_d_timeout=1,
        run_c_mode_tests=False,
        c_mode_test_timeout=1,
        strict=False,
    )
    report = build_report(args)

    assert (tmp_path / "holostuff_metrics.json").exists()
    assert (tmp_path / "holostuff_metrics.md").exists()
    assert report["schema"] == "holostuff-metrics-v1"
    assert "external_baselines" in {section["name"] for section in report["sections"]}


def test_optional_experiment_dependency_failures_are_skips():
    assert _missing_optional_dependency("ModuleNotFoundError: No module named 'sklearn'") == "sklearn"
    assert _missing_optional_dependency("RuntimeError: real bug") is None


def test_c_mode_test_runner_records_selected_passes(tmp_path):
    section = collect_c_mode_tests(
        tmp_path,
        run=True,
        timeout=10,
        tests=(("test_holographic_metrics", "metrics_smoke_pass"),),
        modes=("numpy",),
    )
    metrics = {row["name"]: row for row in section["metrics"]}

    assert metrics["c_mode_tests.numpy.test_holographic_metrics.metrics_smoke_pass.passed"]["value"] is True
    assert metrics["c_mode_tests.numpy.passed_count"]["value"] == 1
