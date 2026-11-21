#!/usr/bin/env python3
"""Print evaluation-count statistics pooled across algorithms and runs per system.

One nonblank CSV data record in *_results.csv counts as one evaluation,
including repeated configurations and initialization records. Counts describe
logged evaluations, not unique configurations or unlogged failed attempts.
Each run has equal weight; mean/median are computed over run counts directly.
"""

from __future__ import annotations

import argparse
import csv
from math import ceil
from pathlib import Path
from statistics import mean, median, stdev


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results_system"
SYSTEM_ORDER = ("mysql", "postgresql", "httpd", "tomcat", "gcc", "clang", "x264", "x265")


def collect_counts(results_dir: Path, fidelity: str = "all") -> dict:
    """Return {system: [(algorithm, run_name, evaluation_count), ...]}."""
    if not results_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    counts = {}
    for system_dir in sorted(results_dir.iterdir()):
        if not system_dir.is_dir():
            continue
        runs = sorted(system_dir.glob("*/run_*"))
        for run_dir in runs:
            if not run_dir.is_dir():
                continue
            if fidelity != "all" and not run_dir.name.endswith(f"_{fidelity}_fidelity"):
                continue
            files = sorted(run_dir.glob("*_results.csv"))
            if len(files) != 1:
                raise ValueError(f"Expected one *_results.csv in {run_dir}, found {len(files)}")
            with files[0].open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader, None)
                if not header:
                    raise ValueError(f"Missing CSV header: {files[0]}")
                count = 0
                for row in reader:
                    if not any(cell.strip() for cell in row):
                        continue
                    if len(row) != len(header) or row == header:
                        raise ValueError(f"Invalid record at line {reader.line_num}: {files[0]}")
                    count += 1
            counts.setdefault(system_dir.name, []).append(
                (run_dir.parent.name, run_dir.name, count)
            )
    if not counts:
        raise ValueError(f"No matching run results found in {results_dir}")
    return counts


def print_summary(counts: dict, robust: bool = False) -> None:
    headers = ("System", "Algorithms", "Runs", "Median evaluations", "Mean evaluations")
    if robust:
        headers += ("SD", "Q1", "Q3", "Trim5% mean", "Trimmed/tail")
    rows = []
    systems = [s for s in SYSTEM_ORDER if s in counts]
    systems += sorted(set(counts) - set(SYSTEM_ORDER))
    for system in systems:
        records = counts[system]
        values = [record[2] for record in records]
        rows.append((system, str(len({r[0] for r in records})), str(len(values)),
                     f"{median(values):.2f}", f"{mean(values):.2f}"))
        if robust:
            ordered = sorted(values)

            def percentile(fraction):
                position = (len(ordered) - 1) * fraction
                lower = int(position)
                return ordered[lower] + (ordered[ceil(position)] - ordered[lower]) * (position - lower)

            # Remove floor(0.05 * n) observations from each tail, uniformly
            # across systems. Quartiles use linear interpolation (type 7).
            k = int(len(values) * 0.05)
            retained = ordered[k:len(ordered) - k] if k else ordered
            sd = f"{stdev(values):.2f}" if len(values) > 1 else "NA"
            rows[-1] += (sd, f"{percentile(0.25):.2f}", f"{percentile(0.75):.2f}",
                         f"{mean(retained):.2f}", str(k))
    widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    for index, row in enumerate([headers] + rows):
        print("  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                        for i, cell in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fidelity", choices=("all", "single", "multi"), default="all",
                        help="Select run-directory fidelity suffix (default: all).")
    parser.add_argument("--robust", action="store_true",
                        help="Also show sample SD, Q1/Q3, and a 5%%-per-tail trimmed mean.")
    args = parser.parse_args()
    try:
        counts = collect_counts(args.results_dir, args.fidelity)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Results: {args.results_dir.resolve()} | Fidelity: {args.fidelity}")
    print("One CSV data record = one logged evaluation; each algorithm/run has equal weight.\n")
    print_summary(counts, robust=args.robust)


if __name__ == "__main__":
    main()
