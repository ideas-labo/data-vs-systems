import argparse
import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union


def count_csv_samples(csv_path: Path) -> int:
    """Count non-empty data rows in a CSV file, excluding the header."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            return 0

        return sum(1 for row in reader if any(cell.strip() for cell in row))


def parse_dataset_file(csv_path: Path) -> tuple:
    system = csv_path.parent.name
    expected_prefix = f"{system}_"
    stem = csv_path.stem

    if stem.startswith(expected_prefix):
        scale = stem[len(expected_prefix):]
    else:
        scale = stem

    return system, scale


def iter_dataset_files(
    datasets_dir: Path,
    system: Optional[str] = None,
    scale: Optional[str] = None,
) -> Iterator[Path]:
    if system and scale:
        yield datasets_dir / system / f"{system}_{scale}.csv"
        return

    system_dirs = [datasets_dir / system] if system else sorted(
        path for path in datasets_dir.iterdir() if path.is_dir()
    )

    for system_dir in system_dirs:
        if not system_dir.exists():
            continue

        pattern = f"{system_dir.name}_{scale}.csv" if scale else f"{system_dir.name}_*.csv"
        yield from sorted(system_dir.glob(pattern))


def collect_sample_counts(
    datasets_dir: Path,
    system: Optional[str] = None,
    scale: Optional[str] = None,
) -> List[Dict[str, Union[str, int]]]:
    rows = []

    for csv_path in iter_dataset_files(datasets_dir, system=system, scale=scale):
        dataset_system, dataset_scale = parse_dataset_file(csv_path)

        if not csv_path.exists():
            rows.append({
                "system": dataset_system,
                "scale": dataset_scale,
                "samples": "MISSING",
                "path": str(csv_path),
            })
            continue

        rows.append({
            "system": dataset_system,
            "scale": dataset_scale,
            "samples": count_csv_samples(csv_path),
            "path": str(csv_path),
        })

    return rows


def print_table(rows: List[Dict[str, Union[str, int]]]) -> None:
    if not rows:
        print("No dataset CSV files found.")
        return

    columns = ("system", "scale", "samples", "path")
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count samples in datasets/{system}/{system}_{scale}.csv files."
    )
    parser.add_argument(
        "--datasets",
        default="../datasets",
        help="datasets directory path, default: datasets",
    )
    parser.add_argument("--system", help="system name, such as mysql or clang")
    parser.add_argument("--scale", help="scale name, such as small, medium, or large")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets).resolve()
    if not datasets_dir.exists():
        raise SystemExit(f"[ERROR] datasets directory does not exist: {datasets_dir}")

    rows = collect_sample_counts(
        datasets_dir=datasets_dir,
        system=args.system,
        scale=args.scale,
    )
    print_table(rows)


if __name__ == "__main__":
    main()
