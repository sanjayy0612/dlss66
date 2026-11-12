"""Pair and validate deterministic Godot shadow captures."""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "dataset" / "raw"


def load(variant):
    records = [json.loads(line) for line in (ROOT / f"manifest_{variant}.jsonl").read_text().splitlines()]
    assert len(records) == int(os.getenv("DLSS66_LAYOUTS", "8")) * int(os.getenv("DLSS66_FRAMES", "12")), (
        variant, len(records)
    )
    by_key = {}
    for record in records:
        key = (record["scene_id"], record["frame_id"])
        assert key not in by_key, f"duplicate {variant} frame {key}"
        assert record["variant"] == variant
        by_key[key] = record
    return by_key


cheap = load("cheap")
reference = load("reference")
assert cheap.keys() == reference.keys(), "missing paired frames"
records = []
differences = []
changed_fractions = []
splits = Counter()
for key in sorted(cheap):
    a_meta, b_meta = cheap[key], reference[key]
    for field in ("seed", "camera_position", "look_at", "effect", "resolution"):
        assert a_meta[field] == b_meta[field], (key, field)
    scene = key[0]
    split = "train" if scene < 6 else "validation" if scene == 6 else "test"
    splits[split] += 1
    a = np.asarray(Image.open(ROOT / a_meta["image"]).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(ROOT / b_meta["image"]).convert("RGB"), dtype=np.int16)
    assert a.shape == b.shape == (144, 256, 3), (key, a.shape, b.shape)
    difference = np.abs(a - b)
    differences.append(float(difference.mean()))
    changed_fractions.append(float(np.mean(np.any(difference > 5, axis=2))))
    records.append({"scene_id": scene, "frame_id": key[1], "split": split,
                    "seed": a_meta["seed"], "camera_position": a_meta["camera_position"],
                    "look_at": a_meta["look_at"], "cheap": a_meta["image"],
                    "reference": b_meta["image"], "effect": a_meta["effect"],
                    "resolution": a_meta["resolution"]})
assert min(differences) > 0, "at least one pair has no visible rendering change"
(ROOT / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
summary = {"pairs": len(records), "scenes": len({r["scene_id"] for r in records}),
           "splits": dict(splits), "mean_abs_rgb_8bit": float(np.mean(differences)),
           "min_pair_mae": min(differences), "max_pair_mae": max(differences),
           "mean_fraction_pixels_changed_gt5": float(np.mean(changed_fractions)),
           "effect": records[0]["effect"]}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("DATASET_VALIDATION", json.dumps(summary))
