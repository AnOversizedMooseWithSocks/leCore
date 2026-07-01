#!/usr/bin/env python3
"""Central metrics spine for holostuff experiments.

The repo has several honest measurement suites already: external baselines,
ablations, stress probes, Path D scripts, and C-kernel evidence. This module
pulls their machine-readable pieces into one JSON + Markdown report, and marks
which experiment families still need fresh caches instead of silently dropping
them.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent

PATH_D_SCRIPTS = {
    "pivot_tree": "experiment_pivot_tree.py",
    "distributed_forward": "experiment_distributed_forward_pass.py",
    "factor_wall": "experiment_factor_wall.py",
    "batch234": "exp_batch_234.py",
    "batchB": "exp_batch_B.py",
    "below_federation": "experiment_below_federation.py",
    "array_router": "experiment_array_router.py",
}

PATH_D_CORE = ("pivot_tree", "distributed_forward", "factor_wall", "batch234", "batchB")
STATUS_ORDER = {"pass": 0, "skip": 1, "warn": 2, "fail": 3}


def _jsonable(value: Any) -> Any:
    """Convert numpy-ish values into plain JSON values without importing numpy."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _metric(
    name: str,
    value: Any,
    *,
    status: str = "pass",
    threshold: str = "",
    unit: str = "",
    details: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "value": _jsonable(value),
        "status": status,
        "threshold": threshold,
        "unit": unit,
        "details": details,
    }


def _section(
    name: str,
    metrics: list[dict[str, Any]],
    *,
    findings: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if not metrics:
        status = "skip"
    else:
        statuses = {m["status"] for m in metrics}
        if "fail" in statuses:
            status = "fail"
        elif "warn" in statuses or ("pass" in statuses and "skip" in statuses):
            status = "warn"
        elif "skip" in statuses:
            status = "skip"
        else:
            status = "pass"
    return {
        "name": name,
        "status": status,
        "metrics": metrics,
        "findings": findings or [],
        "notes": notes or [],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _geomean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0.0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0


def _cliff(xs: list[Any], ys: list[Any], threshold: float) -> int:
    return int(max((x for x, y in zip(xs, ys) if float(y) >= threshold), default=0))


def _dict_series(mapping: dict[str, Any], key: int | str) -> Any:
    return mapping.get(key, mapping.get(str(key)))


def _git_value(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def collect_external_baselines() -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    findings: list[str] = []
    try:
        from benchmarks.bench_compression import compare_compression
        from benchmarks.bench_recall import compare_recall

        compression = compare_compression(Ns=(200, 2000))
        for row in compression:
            ratio = float(row["rd_bits"]) / float(row["int8_zlib_bits"])
            dataset = row["dataset"]
            n = int(row["N"])
            if dataset == "structured":
                status = "pass" if ratio < 1.0 and float(row["rd_cos"]) >= 0.999 else "fail"
                threshold = "rd/int8+zlib < 1 at cosine >= 0.999"
            else:
                status = "pass" if ratio > 1.0 else "warn"
                threshold = "kept negative: random full-rank should favor zlib"
            metrics.append(
                _metric(
                    f"compression.{dataset}.N{n}.rd_to_int8_zlib_bits",
                    ratio,
                    status=status,
                    threshold=threshold,
                    details=(
                        f"rd={row['rd_bits']:.1f} bits/vector, "
                        f"int8+zlib={row['int8_zlib_bits']:.1f}, rd_cos={row['rd_cos']:.4f}"
                    ),
                )
            )

        recall = compare_recall(Ns=(500, 2000), Q=40)
        for row in recall:
            n = int(row["N"])
            cmp_fraction = float(row["forest_cmp"]) / float(row["brute_cmp"])
            metrics.append(
                _metric(
                    f"recall.N{n}.forest_comparison_fraction",
                    cmp_fraction,
                    status="pass" if cmp_fraction < 1.0 else "fail",
                    threshold="forest comparisons/query < brute-force comparisons/query",
                    details=f"forest={row['forest_cmp']} cmp, brute={row['brute_cmp']} cmp",
                )
            )
            metrics.append(
                _metric(
                    f"recall.N{n}.forest_recall1",
                    row["forest_recall1"],
                    status="pass" if float(row["forest_recall1"]) >= 0.95 else "fail",
                    threshold="recall@1 >= 0.95 on smoke-scale harness",
                    details=f"forest@8={row['forest_recall8']:.3f}",
                )
            )
            metrics.append(
                _metric(
                    f"recall.N{n}.wall_time_speedup",
                    row["speedup"],
                    status="pass" if float(row["speedup"]) >= 1.0 else "warn",
                    threshold="kept negative: pure Python forest may lose wall-time to BLAS",
                    details=f"brute={row['brute_us']:.0f}us, forest={row['forest_us']:.0f}us",
                )
            )
        findings.append("External baselines are executable and include kept negatives.")
    except Exception as exc:
        metrics.append(
            _metric(
                "external_baselines.available",
                False,
                status="fail",
                details=f"{type(exc).__name__}: {exc}",
            )
        )
    return _section("external_baselines", metrics, findings=findings)


def _ablation_row_metrics(name: str, h: dict[str, Any], b: dict[str, Any], verdict: str) -> list[dict[str, Any]]:
    slug = _slug(name)
    delta = float(h["mean"]) - float(b["mean"])
    rows = [
        _metric(f"ablation.{slug}.holo_mean", h["mean"], details=verdict),
        _metric(f"ablation.{slug}.baseline_mean", b["mean"], details=verdict),
        _metric(
            f"ablation.{slug}.delta_holo_minus_baseline",
            delta,
            status="pass" if verdict != "skipped" else "skip",
            details=verdict,
        ),
    ]
    if "comparison_fraction" in h:
        rows.append(
            _metric(
                f"ablation.{slug}.comparison_fraction",
                h["comparison_fraction"],
                status="pass" if float(h["comparison_fraction"]) < 0.6 else "warn",
                threshold="forest should use less than 60% of exact comparisons in this row",
                details="scale win despite exact-scan accuracy win",
            )
        )
    return rows


def collect_ablations(full: bool = False) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    findings: list[str] = []
    try:
        from holographic_ablate import (
            ablation_table,
            fdr_verdicts,
            key_value_noisy,
            recall_index,
            verdict,
        )

        if full:
            rows = ablation_table()
        else:
            rows = []
            for name, fn in (
                ("key->value, noisy keys", key_value_noisy),
                ("recall index (forest)", recall_index),
            ):
                h, b, base_name = fn(seeds=range(4))
                rows.append((name, h, b, base_name, verdict(h, b)))

        aug, n_lb, n_surv = fdr_verdicts(rows, alpha=0.1)
        metrics.append(
            _metric(
                "ablation.family.load_bearing_count",
                n_lb,
                details="95% CI verdicts before family-wise FDR control",
            )
        )
        metrics.append(
            _metric(
                "ablation.family.fdr_surviving_load_bearing_count",
                n_surv,
                status="pass" if n_surv <= n_lb else "fail",
                details="BH-Yekutieli alpha=0.1 across the ablation family",
            )
        )
        for name, h, b, base_name, row_verdict, p_value, survives in aug:
            slug = _slug(name)
            if h is None:
                metrics.append(
                    _metric(
                        f"ablation.{slug}.available",
                        False,
                        status="skip",
                        details=f"skipped: {base_name}",
                    )
                )
                continue
            metrics.extend(_ablation_row_metrics(name, h, b, row_verdict))
            metrics.append(
                _metric(
                    f"ablation.{slug}.fdr_survives",
                    survives,
                    status="pass",
                    details=f"p={p_value:.4f}; baseline={base_name}; verdict={row_verdict}",
                )
            )
        findings.append("Fast mode records algebraic load-bearing and forest scale rows.")
        if not full:
            findings.append("Use --full-ablations for Reuters/UDHR/Brown corpus rows.")
    except Exception as exc:
        metrics.append(
            _metric(
                "ablations.available",
                False,
                status="fail",
                details=f"{type(exc).__name__}: {exc}",
            )
        )
    return _section("ablations", metrics, findings=findings)


def collect_c_evidence(summary_path: Path | None = None) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    findings: list[str] = []
    summary_path = summary_path or ROOT / "c" / "build" / "ci-evidence" / "summary.jsonl"
    if not summary_path.exists():
        metrics.append(
            _metric(
                "c_kernel.ci_evidence_present",
                False,
                status="skip",
                details="run `make c-ci-evidence` to populate c/build/ci-evidence/summary.jsonl",
            )
        )
        return _section("c_kernel", metrics, notes=["No C evidence summary found yet."])

    try:
        rows = _read_jsonl(summary_path)
        store = [float(row["store_speedup_c_over_python"]) for row in rows]
        query = [float(row["query_speedup_c_over_python"]) for row in rows]
        acc = [float(row["c_accuracy_median"]) for row in rows]
        metrics.extend(
            [
                _metric(
                    "c_kernel.trace.store_speedup_geomean",
                    _geomean(store),
                    status="pass" if min(store, default=0.0) >= 1.05 else "fail",
                    threshold="every trace dimension >= 1.05x store speedup",
                    details=str(summary_path),
                ),
                _metric(
                    "c_kernel.trace.query_speedup_geomean",
                    _geomean(query),
                    status="pass" if min(query, default=0.0) >= 1.05 else "fail",
                    threshold="every trace dimension >= 1.05x query speedup",
                    details=str(summary_path),
                ),
                _metric(
                    "c_kernel.trace.min_accuracy",
                    min(acc, default=0.0),
                    status="pass" if min(acc, default=0.0) >= 0.99 else "fail",
                    threshold="C trace/action-index accuracy >= 0.99",
                    details=f"{len(rows)} dimensions",
                ),
            ]
        )
        findings.append("C evidence summary was found and folded into the central report.")
    except Exception as exc:
        metrics.append(
            _metric(
                "c_kernel.ci_evidence_parseable",
                False,
                status="fail",
                details=f"{type(exc).__name__}: {exc}",
            )
        )
    return _section("c_kernel", metrics, findings=findings)


def _run_path_d_scripts(cache_dir: Path, mode: str, timeout: int) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = PATH_D_CORE if mode == "core" else tuple(PATH_D_SCRIPTS)
    metrics: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLBACKEND", "Agg")
    exp_dir = ROOT / "path_d" / "experiments"
    for name in names:
        script = exp_dir / PATH_D_SCRIPTS[name]
        log_path = cache_dir / f"{name}.log"
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=cache_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            log_path.write_text(proc.stdout, encoding="utf-8")
            missing_dep = _missing_optional_dependency(proc.stdout)
            status = "pass" if proc.returncode == 0 else ("skip" if missing_dep else "fail")
            details = f"log={log_path}"
            if missing_dep:
                details = f"missing optional dependency {missing_dep}; {details}"
            metrics.append(
                _metric(
                    f"path_d.run.{name}",
                    round(time.perf_counter() - started, 3),
                    status=status,
                    unit="s",
                    details=details,
                )
            )
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(exc.stdout or "", encoding="utf-8")
            metrics.append(
                _metric(
                    f"path_d.run.{name}",
                    timeout,
                    status="fail",
                    unit="s",
                    details=f"timed out; log={log_path}",
                )
            )
    return metrics


def _missing_optional_dependency(output: str) -> str | None:
    match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", output)
    if not match:
        return None
    missing = match.group(1)
    if missing in {"sklearn", "pandas", "matplotlib", "scipy"}:
        return missing
    return None


def _load_cache(cache_dir: Path, filename: str) -> dict[str, Any] | None:
    for path in (
        cache_dir / filename,
        ROOT / filename,
        ROOT / "path_d" / "experiments" / filename,
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _path_d_cache_skip(name: str, filename: str) -> dict[str, Any]:
    return _metric(
        f"path_d.{name}.cache_present",
        False,
        status="skip",
        details=f"missing {filename}; run `make metrics-path-d` or `python holographic_metrics.py --run-path-d`",
    )


def _collect_tree_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data["results"]
    deep = max(rows, key=lambda row: int(row["depth"]))
    exhaustive = float(data["exhaustive"])
    fewer = float(data["K"]) / float(deep["comp_b1"])
    return [
        _metric(
            "path_d.pivot_tree.depth4_top1_gap_to_exhaustive",
            float(deep["top1_b1"]) - exhaustive,
            status="pass" if float(deep["top1_b1"]) >= exhaustive - 0.01 else "fail",
            threshold="greedy top-1 within 0.01 of exhaustive ceiling",
            details=f"top1={deep['top1_b1']:.3f}, exhaustive={exhaustive:.3f}",
        ),
        _metric(
            "path_d.pivot_tree.depth4_beam5_recall",
            deep["rec_b5"],
            status="pass" if float(deep["rec_b5"]) >= 0.95 else "warn",
            threshold="beam-5 true-shard recall >= 0.95",
        ),
        _metric(
            "path_d.pivot_tree.depth4_comparison_reduction",
            fewer,
            status="pass" if fewer >= 10.0 else "warn",
            threshold="at least 10x fewer comparisons than exhaustive scan",
            details=f"comp_b1={deep['comp_b1']:.0f}, K={data['K']}",
        ),
    ]


def _collect_fwd_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    cs = data["Cs"]
    res = data["res"]
    metrics: list[dict[str, Any]] = []
    cliffs: dict[int, int] = {}
    for key in data["Ks"]:
        k = int(key)
        fid = _dict_series(res, k)["fid"]
        cliffs[k] = _cliff(cs, fid, 0.90)
        metrics.append(
            _metric(
                f"path_d.distributed_forward.K{k}.fidelity90_class_cliff",
                cliffs[k],
                status="pass" if cliffs[k] > 0 else "warn",
                threshold="max class count with logit fidelity >= 0.90",
            )
        )
    if 1 in cliffs and max(cliffs) != 1:
        top_k = max(cliffs)
        gain = cliffs[top_k] / max(cliffs[1], 1)
        metrics.append(
            _metric(
                f"path_d.distributed_forward.K{top_k}_capacity_gain_over_single",
                gain,
                status="pass" if gain >= 2.0 else "warn",
                threshold="federation should move the class-fidelity wall",
            )
        )
    return metrics


def _collect_factor_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data["rows"]
    dense_cliff = _cliff([row["F"] for row in rows], [row["dense"] for row in rows], 0.90)
    sbc_cliff = _cliff([row["F"] for row in rows], [row["sbc"] for row in rows], 0.90)
    return [
        _metric(
            "path_d.factor_wall.dense_factor_cliff",
            dense_cliff,
            threshold="max F solved at >= 0.90",
        ),
        _metric(
            "path_d.factor_wall.sbc_factor_cliff",
            sbc_cliff,
            status="pass" if sbc_cliff > dense_cliff else "warn",
            threshold="SBC should push the factorization wall beyond dense",
        ),
        _metric(
            "path_d.factor_wall.sbc_extra_factors",
            sbc_cliff - dense_cliff,
            status="pass" if sbc_cliff - dense_cliff >= 1 else "warn",
        ),
    ]


def _collect_batch234_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    ms, a2 = data["A2"]
    for key, vals in a2.items():
        k = int(key)
        metrics.append(
            _metric(
                f"path_d.bucketA.A2.K{k}.fidelity90_row_cliff",
                _cliff(ms, vals, 0.90),
                details="superposed matmul fidelity vs rows",
            )
        )
    hs, sel, rnk = data["A3"]
    for key, vals in sel.items():
        k = int(key)
        metrics.append(
            _metric(
                f"path_d.bucketA.A3.K{k}.selection95_hypothesis_cliff",
                _cliff(hs, vals, 0.95),
                details=f"rank_corr_at_max={_dict_series(rnk, k)[-1]:.3f}",
            )
        )
    ts, acc = data["A4"]
    for key, vals in acc.items():
        k = int(key)
        metrics.append(
            _metric(
                f"path_d.bucketA.A4.K{k}.recall90_sequence_cliff",
                _cliff(ts, vals, 0.90),
                details="sequence length with >=90% symbol recall",
            )
        )
    return metrics


def _collect_batchB_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    ns, mono, fed = data["A5"]
    max_gap = max(abs(float(m) - float(f)) for m, f in zip(mono, fed))
    metrics.append(
        _metric(
            "path_d.bucketA.A5.fixed_total_dim_max_corr_gap",
            max_gap,
            status="pass" if max_gap <= 0.15 else "warn",
            threshold="federated archive should conserve quality at fixed total dimension",
            details=f"N range {min(ns)}..{max(ns)}",
        )
    )
    ks, ranges, acc = data["A6"]
    cliffs = {int(k): _cliff(ks, vals, 0.95) for k, vals in acc.items()}
    for k, cliff in cliffs.items():
        metrics.append(
            _metric(
                f"path_d.bucketA.A6.K{k}.roundtrip95_moduli_cliff",
                cliff,
                details="CRT residue round-trip >=95%",
            )
        )
    if 1 in cliffs and max(cliffs) != 1:
        top = max(cliffs)
        metrics.append(
            _metric(
                f"path_d.bucketA.A6.K{top}_moduli_gain_over_single",
                cliffs[top] / max(cliffs[1], 1),
                status="pass" if cliffs[top] > cliffs[1] else "warn",
                threshold="federated residue range should exceed single-vector range",
            )
        )
    return metrics


def _collect_below_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    pc = {int(k): float(v) for k, v in data["pc"].items()}
    spread = (max(pc.values()) - min(pc.values())) / max(pc.values())
    metrics = [
        _metric(
            "path_d.below_federation.partition_capacity_relative_spread",
            spread,
            status="pass" if spread <= 0.35 else "warn",
            threshold="fixed-D partition capacity should be roughly conserved",
            details=f"capacities={pc}",
        )
    ]
    for scale in ("block", "array"):
        curves = data[scale]
        for parity in ("1", "2"):
            curve = curves.get(parity, curves.get(int(parity)))
            if curve is None:
                continue
            survived = float(curve[int(parity)])
            metrics.append(
                _metric(
                    f"path_d.below_federation.{scale}.parity{parity}_recall_at_{parity}_losses",
                    survived,
                    status="pass" if survived >= 0.85 else "warn",
                    threshold="M parity reconstructs M lost units",
                )
            )
    return metrics


def _collect_router_cache(data: dict[str, Any]) -> list[dict[str, Any]]:
    part1 = data["part1"]
    last = max(part1, key=lambda row: int(row[0]))
    k, directory, routed, broadcast, dt, rt, bt = last
    metrics = [
        _metric(
            "path_d.array_router.maxK_routed_recall",
            routed,
            status="pass" if float(routed) >= 0.90 else "warn",
            threshold="routed recall should stay high at max shard count",
            details=f"K={k}, directory={directory:.3f}",
        ),
        _metric(
            "path_d.array_router.maxK_directory_to_routed_time_ratio",
            float(dt) / max(float(rt), 1e-12),
            status="pass" if float(rt) <= float(dt) * 4 else "warn",
            details="wall-time sanity for routed lookup",
        ),
    ]
    part2 = data["part2"]
    one = next(row for row in part2 if str(row[0]).startswith("1-level"))
    candidates = [row for row in part2 if not str(row[0]).startswith("1-level")]
    best = max(candidates, key=lambda row: (float(row[1]), -int(row[2]))) if candidates else one
    metrics.append(
        _metric(
            "path_d.array_router.best_2level_comparison_fraction",
            int(best[2]) / int(one[2]),
            status="pass" if int(best[2]) < int(one[2]) else "warn",
            threshold="2-level route should reduce routing comparisons when recall holds",
            details=f"{best[0]} recall={float(best[1]):.3f}; 1-level recall={float(one[1]):.3f}",
        )
    )
    return metrics


def collect_path_d(output_dir: Path, run_mode: str | None = None, path_d_timeout: int = 240) -> dict[str, Any]:
    cache_dir = output_dir / "path_d-cache"
    metrics: list[dict[str, Any]] = []
    findings = ["Path D metrics are read from cache JSON emitted by the experiment scripts."]

    if run_mode:
        metrics.extend(_run_path_d_scripts(cache_dir, run_mode, path_d_timeout))

    cache_collectors = (
        ("pivot_tree", "_tree_cache.json", _collect_tree_cache),
        ("distributed_forward", "_fwd_cache.json", _collect_fwd_cache),
        ("factor_wall", "_factor_cache.json", _collect_factor_cache),
        ("bucketA_234", "_batch234_cache.json", _collect_batch234_cache),
        ("bucketA_B", "_batchB_cache.json", _collect_batchB_cache),
        ("below_federation", "_below_cache.json", _collect_below_cache),
        ("array_router", "_router_cache.json", _collect_router_cache),
    )
    for name, filename, collector in cache_collectors:
        data = _load_cache(cache_dir, filename)
        if data is None:
            metrics.append(_path_d_cache_skip(name, filename))
            continue
        try:
            metrics.extend(collector(data))
        except Exception as exc:
            metrics.append(
                _metric(
                    f"path_d.{name}.cache_parseable",
                    False,
                    status="fail",
                    details=f"{filename}: {type(exc).__name__}: {exc}",
                )
            )

    return _section("path_d", metrics, findings=findings)


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def collect_stress(output_dir: Path) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    findings: list[str] = []
    try:
        import stress_holographic as stress

        stress.FINDINGS.clear()
        with _pushd(output_dir / "stress"):
            sep = stress.stress_separability()
            stale, committed = stress.stress_disappearance()
            scaling = stress.stress_scaling()
            conformal = stress.stress_conformal_shift()
            abrupt, gradual = stress.stress_gradual_drift()
            pred = stress.stress_predictive_sensitivity()
            scalar = stress.stress_scalar_range()
        breakpt = next(
            (
                float(row.cluster_std)
                for row in sep.itertuples()
                if float(row.Emergent) < 0.8
            ),
            None,
        )
        metrics.extend(
            [
                _metric(
                    "stress.separability.emergent_ari_breakpoint",
                    breakpt if breakpt is not None else ">max",
                    status="pass" if breakpt is None or breakpt >= 8 else "warn",
                    threshold="Emergent ARI should stay >=0.8 through moderate overlap",
                ),
                _metric(
                    "stress.disappearance.stale_concepts",
                    stale,
                    status="pass" if int(stale) == 0 else "warn",
                    threshold="dead categories should retire",
                    details=f"committed={committed}",
                ),
                _metric(
                    "stress.scaling.max_cluster_count_error",
                    float(scaling["k_error"].max()),
                    status="pass" if float(scaling["k_error"].max()) <= 3 else "warn",
                ),
                _metric(
                    "stress.conformal.no_shift_coverage",
                    float(conformal.iloc[0]["empirical"]),
                    status="pass" if float(conformal.iloc[0]["empirical"]) >= 0.85 else "fail",
                    threshold="control coverage near 0.90",
                ),
                _metric(
                    "stress.drift.abrupt_layers",
                    abrupt,
                    details=f"gradual_layers={gradual}",
                ),
                _metric(
                    "stress.predictive.reliable_detection_floor",
                    next(
                        (
                            float(row.magnitude)
                            for row in pred.itertuples()
                            if float(row.detection) >= 0.8
                        ),
                        None,
                    ),
                    status="pass",
                    details=f"max_false_alarms={float(pred['false_alarms'].max()):.3f}",
                ),
                _metric(
                    "stress.scalar_range.max_relative_decode_error",
                    float(scalar["rel_decode_error"].max()),
                    status="pass" if float(scalar["rel_decode_error"].max()) <= 0.02 else "warn",
                ),
            ]
        )
        findings.append("Full stress probes ran and wrote plots under the metrics output directory.")
    except Exception as exc:
        metrics.append(
            _metric(
                "stress.available",
                False,
                status="skip",
                details=f"{type(exc).__name__}: {exc}",
            )
        )
    return _section("stress", metrics, findings=findings)


def _summary(sections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in STATUS_ORDER}
    for section in sections:
        for metric in section["metrics"]:
            counts[metric["status"]] += 1
    status = max((section["status"] for section in sections), key=lambda s: STATUS_ORDER[s])
    return {"status": status, "counts": counts}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        collect_external_baselines(),
        collect_ablations(full=args.full_ablations),
        collect_c_evidence(),
        collect_path_d(output_dir, run_mode=args.run_path_d, path_d_timeout=args.path_d_timeout),
    ]
    if args.include_stress:
        sections.append(collect_stress(output_dir))

    report = {
        "schema": "holostuff-metrics-v1",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "git": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "sections": sections,
    }
    report["summary"] = _summary(sections)

    json_path = output_dir / "holostuff_metrics.json"
    md_path = output_dir / "holostuff_metrics.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(_short_console_summary(report))
    return report


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Holostuff Metrics",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Git: `{report['git']['branch']}` `{report['git']['commit']}` dirty={report['git']['dirty']}",
        f"- Status: **{report['summary']['status']}**",
        "",
        "## Section Summary",
        "",
        "| section | status | pass | warn | fail | skip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for section in report["sections"]:
        counts = {status: 0 for status in STATUS_ORDER}
        for metric in section["metrics"]:
            counts[metric["status"]] += 1
        lines.append(
            f"| {section['name']} | {section['status']} | {counts['pass']} | "
            f"{counts['warn']} | {counts['fail']} | {counts['skip']} |"
        )

    for section in report["sections"]:
        lines.extend(["", f"## {section['name']}", ""])
        for finding in section["findings"]:
            lines.append(f"- {finding}")
        for note in section["notes"]:
            lines.append(f"- {note}")
        lines.extend(["", "| metric | value | status | threshold/details |", "|---|---:|---:|---|"])
        for metric in section["metrics"]:
            details = metric["threshold"] or metric["details"]
            if metric["threshold"] and metric["details"]:
                details = f"{metric['threshold']}; {metric['details']}"
            lines.append(
                f"| `{metric['name']}` | {_format_value(metric['value'])} | "
                f"{metric['status']} | {details} |"
            )
    return "\n".join(lines) + "\n"


def _short_console_summary(report: dict[str, Any]) -> str:
    parts = [
        f"{section['name']}={section['status']}"
        for section in report["sections"]
    ]
    counts = report["summary"]["counts"]
    return (
        f"Status {report['summary']['status']} "
        f"(pass={counts['pass']} warn={counts['warn']} fail={counts['fail']} skip={counts['skip']}): "
        + ", ".join(parts)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument(
        "--full-ablations",
        action="store_true",
        help="run corpus-gated ablations in addition to the fast algebraic rows",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="run the full stress suite and include stress metrics",
    )
    parser.add_argument(
        "--run-path-d",
        choices=("core", "all"),
        default=None,
        nargs="?",
        const="core",
        help="regenerate Path D cache JSON before collecting it",
    )
    parser.add_argument("--path-d-timeout", type=int, default=240)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any collected metric fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.strict and report["summary"]["counts"]["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
