"""Combine local paired datasets without copying image and geometry files."""

import argparse
import json
import os
from pathlib import Path


def combine(output: Path, sources):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for root, source_split, target_split in sources:
        root = Path(root)
        prefix = Path(os.path.relpath(root, output))
        for line in (root / "manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["split"] != source_split:
                continue
            row = dict(row)
            row["split"] = target_split
            for key in ("cheap", "reference", "geometry"):
                row[key] = str(prefix / row[key])
            rows.append(row)
    if not rows:
        raise ValueError("no source rows matched the requested splits")
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", nargs=3, action="append", required=True,
                        metavar=("ROOT", "SOURCE_SPLIT", "TARGET_SPLIT"))
    args = parser.parse_args()
    print(combine(args.output, args.source))
