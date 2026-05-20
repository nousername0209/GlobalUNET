from __future__ import annotations

import argparse
import csv
from pathlib import Path


def summarize(results_dir: Path, output_csv: Path) -> list[dict[str, str]]:
    metric_files = sorted(results_dir.glob("*_seed*/metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics.csv files found under {results_dir}")

    rows = []
    for path in metric_files:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["run_dir"] = str(path.parent)
                rows.append(row)

    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", ""), row.get("seed", "")))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine experiment metrics into one CSV.")
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-csv", type=Path, default=Path("runs/summary.csv"))
    args = parser.parse_args()
    summary = summarize(args.results_dir, args.output_csv)
    for row in summary:
        print(row)
    print(f"\nSaved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
