#!/usr/bin/env python3
"""Build CI-visible evidence that the C trace kernel beats NumPy.

This script is intentionally focused on the scalar C path because GitHub's
Linux runners do not have Apple's Accelerate framework. It gates the trace
memory workload, where the C kernel has a portable architectural advantage, and
records bind_fixed measurements as supporting data without treating those
platform-sensitive results as a merge blocker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
C_DIR = BENCH_DIR.parent
ROOT = C_DIR.parent

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_trace  # noqa: E402


def _split_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _shlib_name() -> str:
    if sys.platform == "darwin":
        return "libholoc.dylib"
    return "libholoc.so"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    return result.stdout


def _json_lines(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            rows.append(json.loads(stripped))
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _build_scalar_kernel() -> tuple[Path, Path]:
    env = os.environ.copy()
    env["HOLO_USE_ACCELERATE"] = "0"
    env["PYTHON"] = sys.executable
    binary = C_DIR / "build" / "scalar" / "bench_trace"
    shlib = C_DIR / "build" / "scalar" / _shlib_name()
    _run(
        [
            "make",
            "-C",
            str(C_DIR),
            "test",
            "shared",
            str(binary.relative_to(C_DIR)),
            "HOLO_USE_ACCELERATE=0",
            f"PYTHON={sys.executable}",
        ],
        env=env,
    )
    return binary, shlib


def _trace_evidence(
    binary: Path,
    dims: list[int],
    pairs: int,
    actions: int,
    queries: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for dim in dims:
        py_rows = []
        c_rows = []
        for repeat in range(repeats):
            py = bench_trace.python_run(dim, pairs, actions, queries)
            c = bench_trace.c_run(binary, dim, pairs, actions, queries)
            py["repeat"] = repeat
            c["repeat"] = repeat
            py_rows.append(py)
            c_rows.append(c)
            raw_rows.append({"benchmark": "trace", **py})
            raw_rows.append({"benchmark": "trace", **c})

        py_store = _median(py_rows, "stores_per_second")
        c_store = _median(c_rows, "stores_per_second")
        py_query = _median(py_rows, "queries_per_second")
        c_query = _median(c_rows, "queries_per_second")
        summaries.append(
            {
                "benchmark": "trace",
                "runtime": "summary",
                "c_runtime": c_rows[0]["runtime"],
                "dim": dim,
                "pairs": pairs,
                "actions": actions,
                "queries": queries,
                "repeats": repeats,
                "python_store_per_second_median": py_store,
                "c_store_per_second_median": c_store,
                "store_speedup_c_over_python": c_store / py_store if py_store else 0.0,
                "python_query_per_second_median": py_query,
                "c_query_per_second_median": c_query,
                "query_speedup_c_over_python": c_query / py_query if py_query else 0.0,
                "python_accuracy_median": _median(py_rows, "accuracy"),
                "c_accuracy_median": _median(c_rows, "accuracy"),
            }
        )

    return raw_rows, summaries


def _bind_fixed_evidence(
    shlib: Path,
    dims: list[int],
    rows: list[int],
    loops: int,
    repeats: int,
) -> list[dict[str, Any]]:
    env = os.environ.copy()
    env["HOLOSTUFF_C_LIB"] = str(shlib)
    env.pop("HOLOSTUFF_USE_C", None)
    output = _run(
        [
            sys.executable,
            str(BENCH_DIR / "bench_bind_fixed.py"),
            "--summary",
            "--dims",
            ",".join(str(dim) for dim in dims),
            "--rows",
            ",".join(str(row) for row in rows),
            "--loops",
            str(loops),
            "--repeats",
            str(repeats),
        ],
        env=env,
    )
    return [{"benchmark": "bind_fixed", **row} for row in _json_lines(output)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown(
    path: Path,
    trace_summaries: list[dict[str, Any]],
    bind_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    store_speedups = [float(row["store_speedup_c_over_python"]) for row in trace_summaries]
    query_speedups = [float(row["query_speedup_c_over_python"]) for row in trace_summaries]

    lines = [
        "# C Kernel CI Evidence",
        "",
        "Scalar C build, C unit tests, and repeated C-vs-NumPy benchmarks ran in CI.",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- NumPy: `{bench_trace.np.__version__}`",
        f"- Platform: `{platform.platform()}`",
        f"- Machine: `{platform.machine()}`",
        f"- Git SHA: `{os.environ.get('GITHUB_SHA', 'local')}`",
        "",
        "## Trace Kernel Gate",
        "",
        (
            f"Gate: every measured trace dimension must reach at least "
            f"{args.min_store_speedup:.2f}x store speedup, "
            f"{args.min_query_speedup:.2f}x query speedup, and "
            f"{args.min_accuracy:.2f} C accuracy."
        ),
        "",
        "| dim | C runtime | store speedup | query speedup | C accuracy | NumPy accuracy |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in trace_summaries:
        lines.append(
            "| {dim} | `{runtime}` | {store}x | {query}x | {c_acc} | {py_acc} |".format(
                dim=row["dim"],
                runtime=row["c_runtime"],
                store=_format_float(float(row["store_speedup_c_over_python"])),
                query=_format_float(float(row["query_speedup_c_over_python"])),
                c_acc=_format_float(float(row["c_accuracy_median"])),
                py_acc=_format_float(float(row["python_accuracy_median"])),
            )
        )

    lines.extend(
        [
            "",
            (
                f"Geomean speedup across gated dimensions: "
                f"{_format_float(_geomean(store_speedups))}x store, "
                f"{_format_float(_geomean(query_speedups))}x query."
            ),
            "",
            "## bind_fixed Supporting Evidence",
            "",
            "These rows are recorded for reviewer visibility but are not a CI gate; scalar Linux results are platform-sensitive here.",
            "",
            "| dim | rows | C path | speedup | max abs error |",
            "| ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in bind_rows:
        lines.append(
            "| {dim} | {rows} | {path} | {speedup}x | {err:.3e} |".format(
                dim=row["dim"],
                rows=row["rows"],
                path="yes" if row["c_path_expected"] else "fallback",
                speedup=_format_float(float(row["speedup_c_over_python"])),
                err=float(row["max_abs"]),
            )
        )

    lines.extend(["", "## Result", ""])
    if failures:
        lines.append("FAIL")
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("PASS: the scalar C trace kernel beat NumPy on every gated dimension.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=C_DIR / "build" / "ci-evidence")
    parser.add_argument("--trace-dims", default="128,256,512")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--actions", type=int, default=8)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-store-speedup", type=float, default=1.05)
    parser.add_argument("--min-query-speedup", type=float, default=1.05)
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--bind-fixed-dims", default="128,256,512")
    parser.add_argument("--bind-fixed-rows", default="1,8,32")
    parser.add_argument("--bind-fixed-loops", type=int, default=50)
    parser.add_argument("--bind-fixed-repeats", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary, shlib = _build_scalar_kernel()

    trace_raw, trace_summaries = _trace_evidence(
        binary=binary,
        dims=_split_ints(args.trace_dims),
        pairs=args.pairs,
        actions=args.actions,
        queries=args.queries,
        repeats=args.repeats,
    )
    bind_rows = _bind_fixed_evidence(
        shlib=shlib,
        dims=_split_ints(args.bind_fixed_dims),
        rows=_split_ints(args.bind_fixed_rows),
        loops=args.bind_fixed_loops,
        repeats=args.bind_fixed_repeats,
    )

    failures: list[str] = []
    for row in trace_summaries:
        dim = row["dim"]
        store_speedup = float(row["store_speedup_c_over_python"])
        query_speedup = float(row["query_speedup_c_over_python"])
        c_accuracy = float(row["c_accuracy_median"])
        if store_speedup < args.min_store_speedup:
            failures.append(
                f"dim {dim}: store speedup {store_speedup:.3f}x below {args.min_store_speedup:.3f}x"
            )
        if query_speedup < args.min_query_speedup:
            failures.append(
                f"dim {dim}: query speedup {query_speedup:.3f}x below {args.min_query_speedup:.3f}x"
            )
        if c_accuracy < args.min_accuracy:
            failures.append(f"dim {dim}: C accuracy {c_accuracy:.3f} below {args.min_accuracy:.3f}")

    raw_path = args.output_dir / "raw.jsonl"
    summary_path = args.output_dir / "summary.jsonl"
    report_path = args.output_dir / "report.md"
    _write_jsonl(raw_path, trace_raw + bind_rows)
    _write_jsonl(summary_path, trace_summaries)
    _write_markdown(report_path, trace_summaries, bind_rows, args, failures)

    print(f"Wrote raw evidence: {raw_path}")
    print(f"Wrote summary evidence: {summary_path}")
    print(f"Wrote Markdown report: {report_path}")
    print(report_path.read_text(encoding="utf-8"))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
