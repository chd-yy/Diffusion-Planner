#!/usr/bin/env python3
"""Validate and merge runner reports produced by chunked Test14 evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


def load_benchmark(manifest_path: Path, benchmark: str) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        return manifest["benchmarks"][benchmark]
    except KeyError as error:
        raise KeyError(f"Unknown benchmark {benchmark!r}") from error


def expected_chunk(benchmark: Dict[str, object], index: int) -> Dict[str, object]:
    chunks = benchmark["chunks"]
    if index < 0 or index >= len(chunks):
        raise IndexError(f"Chunk index {index} outside [0, {len(chunks)})")
    return chunks[index]


def validate_report(path: Path, expected_tokens: Set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = pd.read_parquet(path)
    required = {"scenario_name", "succeeded"}
    if not required.issubset(report.columns):
        raise RuntimeError(f"Missing columns in {path}: {sorted(required - set(report.columns))}")
    observed = set(report["scenario_name"].astype(str))
    if len(report) != len(expected_tokens) or observed != expected_tokens:
        raise RuntimeError(
            f"Runner coverage mismatch in {path}: rows={len(report)}, "
            f"unique={len(observed)}, expected={len(expected_tokens)}"
        )
    if not report["succeeded"].astype(bool).all():
        failed = report.loc[~report["succeeded"].astype(bool), "scenario_name"].tolist()
        raise RuntimeError(f"Failed simulations in {path}: {failed}")
    return report


def validate_aggregator(run_dir: Path, expected_tokens: Set[str]) -> Path:
    files = sorted((run_dir / "aggregator_metric").glob("*.parquet"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one aggregator parquet in {run_dir}, found {len(files)}")
    metrics = pd.read_parquet(files[0])
    scenarios = metrics[metrics["log_name"].notna()].copy()
    observed = set(scenarios["scenario"].astype(str))
    if len(scenarios) != len(expected_tokens) or observed != expected_tokens:
        raise RuntimeError(
            f"Metric coverage mismatch in {files[0]}: rows={len(scenarios)}, "
            f"unique={len(observed)}, expected={len(expected_tokens)}"
        )
    return files[0]


def expected_runner_tokens(benchmark: Dict[str, object]) -> Set[str]:
    """Return all tokens with recoverable runner timing rows."""

    return set(benchmark.get("runner_tokens", benchmark["tokens"]))


def scenario_rows(path: Path) -> pd.DataFrame:
    """Read only real scenario rows from one NuPlan aggregator parquet."""

    metrics = pd.read_parquet(path)
    required = {"scenario", "log_name"}
    if not required.issubset(metrics.columns):
        raise RuntimeError(f"Missing aggregator columns in {path}: {sorted(required - set(metrics.columns))}")
    return metrics[metrics["log_name"].notna()].copy()


def validate_chunk_aggregator(
    path: Path, benchmark: Dict[str, object], chunk: Dict[str, object]
) -> pd.DataFrame:
    """Validate one archived aggregator, allowing prior recovered metric rows."""

    if not path.is_file():
        raise FileNotFoundError(path)
    scenarios = scenario_rows(path)
    observed = set(scenarios["scenario"].astype(str))
    required = set(chunk["tokens"])
    allowed = required | set(benchmark.get("recovered_metric_tokens", []))
    if not required.issubset(observed) or not observed.issubset(allowed):
        raise RuntimeError(
            f"Chunk aggregator coverage mismatch in {path}: rows={len(scenarios)}, "
            f"unique={len(observed)}, required={len(required)}, allowed={len(allowed)}"
        )
    if len(scenarios) != len(observed):
        raise RuntimeError(f"Duplicate scenario rows in chunk aggregator {path}")
    return scenarios


def validate_chunk(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.manifest, args.benchmark)
    chunk = expected_chunk(benchmark, args.chunk_index)
    validate_report(args.report, set(chunk["tokens"]))
    print(f"Validated {args.benchmark} chunk {args.chunk_index}: {chunk['count']} scenarios")


def archived_paths(run_dir: Path, index: int) -> tuple[Path, Path]:
    return (
        run_dir / "chunk_runner_reports" / f"chunk-{index:03d}.parquet",
        run_dir / "chunk_aggregator_metrics" / f"chunk-{index:03d}.parquet",
    )


def validate_chunk_artifacts(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.manifest, args.benchmark)
    chunk = expected_chunk(benchmark, args.chunk_index)
    report_path, aggregator_path = archived_paths(args.run_dir, args.chunk_index)
    validate_report(report_path, set(chunk["tokens"]))
    validate_chunk_aggregator(aggregator_path, benchmark, chunk)
    print(f"Validated archived {args.benchmark} chunk {args.chunk_index}")


def archive_chunk(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.manifest, args.benchmark)
    chunk = expected_chunk(benchmark, args.chunk_index)
    validate_report(args.report, set(chunk["tokens"]))
    candidates = []
    for path in sorted((args.run_dir / "aggregator_metric").glob("*.parquet")):
        try:
            rows = validate_chunk_aggregator(path, benchmark, chunk)
        except RuntimeError:
            continue
        candidates.append((len(rows), path.stat().st_mtime_ns, path))
    if not candidates:
        raise RuntimeError(
            f"No aggregator parquet covers {args.benchmark} chunk {args.chunk_index}"
        )
    # Prefer the candidate with the fewest extra recovery rows, then the newest file.
    _, _, selected = sorted(candidates, key=lambda item: (item[0], -item[1]))[0]
    report_path, aggregator_path = archived_paths(args.run_dir, args.chunk_index)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aggregator_path.parent.mkdir(parents=True, exist_ok=True)
    if args.report.resolve() != report_path.resolve():
        shutil.copy2(args.report, report_path)
    shutil.copy2(selected, aggregator_path)
    validate_report(report_path, set(chunk["tokens"]))
    validate_chunk_aggregator(aggregator_path, benchmark, chunk)
    print(
        f"Archived {args.benchmark} chunk {args.chunk_index}: "
        f"report={report_path.name}, aggregator_source={selected.name}"
    )


def merge(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.manifest, args.benchmark)
    reports: List[pd.DataFrame] = []
    metric_parts: List[pd.DataFrame] = []
    for chunk in benchmark["chunks"]:
        report_path, aggregator_path = archived_paths(args.run_dir, chunk["index"])
        reports.append(validate_report(report_path, set(chunk["tokens"])))
        metric_parts.append(validate_chunk_aggregator(aggregator_path, benchmark, chunk))
    merged = pd.concat(reports, ignore_index=True)
    expected_tokens = expected_runner_tokens(benchmark)
    observed = set(merged["scenario_name"].astype(str))
    if len(merged) != len(expected_tokens) or observed != expected_tokens:
        raise RuntimeError(
            f"Merged runner coverage mismatch: rows={len(merged)}, "
            f"unique={len(observed)}, expected={len(expected_tokens)}"
        )
    output = args.run_dir / "runner_report.parquet"
    temporary = output.with_suffix(".merge.tmp.parquet")
    merged.to_parquet(temporary)
    temporary.replace(output)
    metric_tokens = set(benchmark["tokens"])
    merged_metrics = pd.concat(metric_parts, ignore_index=True)
    metric_observed = merged_metrics["scenario"].astype(str)
    if len(merged_metrics) != len(metric_tokens) or set(metric_observed) != metric_tokens:
        duplicates = sorted(metric_observed[metric_observed.duplicated()].unique().tolist())
        raise RuntimeError(
            f"Merged metric coverage mismatch: rows={len(merged_metrics)}, "
            f"unique={metric_observed.nunique()}, expected={len(metric_tokens)}, "
            f"duplicates={duplicates}"
        )
    aggregator_root = args.run_dir / "aggregator_metric"
    originals_root = aggregator_root / "chunk_originals"
    originals_root.mkdir(parents=True, exist_ok=True)
    for original in sorted(aggregator_root.glob("*.parquet")):
        destination = originals_root / original.name
        if destination.exists():
            destination = originals_root / f"{original.stem}-{original.stat().st_mtime_ns}.parquet"
        original.replace(destination)
    aggregator = aggregator_root / "chunk_merged_scenario_metrics.parquet"
    temporary_aggregator = aggregator.with_suffix(".merge.tmp.parquet")
    merged_metrics.to_parquet(temporary_aggregator)
    temporary_aggregator.replace(aggregator)
    validate_aggregator(args.run_dir, metric_tokens)
    validation = {
        "benchmark": args.benchmark,
        "scenario_count": len(metric_tokens),
        "chunk_count": len(benchmark["chunks"]),
        "all_succeeded": True,
        "runner_report_count": len(expected_tokens),
        "runner_report_complete": expected_tokens == metric_tokens,
        "metric_only_recovered_scenarios": len(metric_tokens - expected_tokens),
        "runner_report": str(output.resolve()),
        "aggregator_metric": str(aggregator.resolve()),
    }
    (args.run_dir / "chunk_merge_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))


def validate_run(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.manifest, args.benchmark)
    metric_tokens = set(benchmark["tokens"])
    runner_tokens = expected_runner_tokens(benchmark)
    validate_report(args.run_dir / "runner_report.parquet", runner_tokens)
    aggregator = validate_aggregator(args.run_dir, metric_tokens)
    print(
        f"Validated complete {args.benchmark} metrics: {len(metric_tokens)} scenarios, "
        f"runner_rows={len(runner_tokens)}, aggregator={aggregator.name}"
    )


def list_chunks(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.manifest, args.benchmark)
    for chunk in benchmark["chunks"]:
        print(f"{chunk['index']}\t{chunk['filter']}\t{chunk['count']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True, type=Path)
    common.add_argument("--benchmark", required=True)

    chunk = subparsers.add_parser("validate-chunk", parents=[common])
    chunk.add_argument("--chunk-index", required=True, type=int)
    chunk.add_argument("--report", required=True, type=Path)
    chunk.set_defaults(handler=validate_chunk)

    artifacts = subparsers.add_parser("validate-chunk-artifacts", parents=[common])
    artifacts.add_argument("--chunk-index", required=True, type=int)
    artifacts.add_argument("--run-dir", required=True, type=Path)
    artifacts.set_defaults(handler=validate_chunk_artifacts)

    archive = subparsers.add_parser("archive-chunk", parents=[common])
    archive.add_argument("--chunk-index", required=True, type=int)
    archive.add_argument("--run-dir", required=True, type=Path)
    archive.add_argument("--report", required=True, type=Path)
    archive.set_defaults(handler=archive_chunk)

    merge_parser = subparsers.add_parser("merge", parents=[common])
    merge_parser.add_argument("--run-dir", required=True, type=Path)
    merge_parser.set_defaults(handler=merge)

    run = subparsers.add_parser("validate-run", parents=[common])
    run.add_argument("--run-dir", required=True, type=Path)
    run.set_defaults(handler=validate_run)

    listing = subparsers.add_parser("list-chunks", parents=[common])
    listing.set_defaults(handler=list_chunks)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
