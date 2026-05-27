from __future__ import annotations

import argparse
import csv
from pathlib import Path


def collect(results_root: Path, output_csv: Path, output_md: Path) -> list[dict[str, str]]:
    rows = []
    for metrics_path in sorted(results_root.glob("*/metrics.csv")):
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["run_dir"] = str(metrics_path.parent)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {results_root}")
    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", ""), row.get("deep_supervision", "")))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_md.write_text(markdown_table(rows), encoding="utf-8")
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = ["dataset", "model", "deep_supervision", "miou", "dice_f1", "pixel_accuracy", "parameter_count", "inference_latency_ms_per_image", "gpu_memory_mb"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all experiment metrics into CSV and Markdown tables.")
    parser.add_argument("--results-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-csv", type=Path, default=Path("runs/summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/summary.md"))
    args = parser.parse_args()
    rows = collect(args.results_root, args.output_csv, args.output_md)
    print(markdown_table(rows))
    print(f"Saved {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
