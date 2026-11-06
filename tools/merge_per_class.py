"""Merge per-class AP CSV files into one wide DIOR comparison table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_result(path, method):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    summary = next((r for r in rows if r.get("category") == "mAP"), None)
    if summary is None:
        raise ValueError(f"{path} has no mAP summary row")
    values = {r["category"]: r["AP"] for r in rows if r.get("category") != "mAP"}
    return {"method": method, "source": "measured", "AP": summary["AP"], "AP50": summary["AP50"], **values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=CSV")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records, categories = [], []
    for item in args.input:
        if "=" not in item:
            parser.error(f"--input must be METHOD=CSV, got {item!r}")
        method, path = item.split("=", 1)
        record = read_result(Path(path), method)
        records.append(record)
        categories.extend(k for k in record if k not in {"method", "source", "AP", "AP50"} and k not in categories)
    fields = ["method", "source", "AP", "AP50", *categories]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.output}: {len(records)} methods, {len(categories)} categories")


if __name__ == "__main__":
    main()
