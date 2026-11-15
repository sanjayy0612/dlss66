"""Summarize Godot GPU intervals from an exported Xcode Metal System Trace.

This is a diagnostic, not an end-to-end frame latency measurement. Xcode's
`metal-gpu-intervals` table records channel activity; overlapping intervals
are merged before estimating busy time for each Godot frame.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def resolve(element, ids):
    while element is not None and element.get("ref"):
        element = ids.get(element.get("ref"))
    return element


def merged_duration(intervals):
    intervals.sort()
    start, end = intervals[0]
    total = 0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def summarize(values):
    return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95))}


def run(toc_path: Path, intervals_path: Path, log_path: Path):
    toc = ET.parse(toc_path).getroot()
    trace_start = datetime.fromisoformat(toc.findtext(".//start-date")).timestamp()
    phases = []
    for line in log_path.read_text().splitlines():
        if not line.startswith("ATLAS_BENCHMARK"):
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        phases.append({"atlas": int(values["size"]),
                       "pcss_angular_deg": float(values.get("pcss_angular_deg", "0")),
                       "ssil_enabled": bool(int(values.get("ssil_enabled", "0"))),
                       "start": float(values["phase_start_unix_s"]),
                       "end": float(values["phase_end_unix_s"])})
    if not phases:
        raise ValueError("benchmark log has no phase timestamps")

    root = ET.parse(intervals_path).getroot()
    ids = {node.get("id"): node for node in root.iter() if node.get("id")}
    frames = defaultdict(list)
    for row in root.findall(".//row"):
        process = resolve(row.find("process"), ids)
        if process is None or not process.get("fmt", "").startswith("Godot ("):
            continue
        frame = resolve(row.find("gpu-frame-number"), ids)
        start = resolve(row.find("start-time"), ids)
        duration = resolve(row.find("duration"), ids)
        if frame is None or start is None or duration is None or not frame.text:
            continue
        frame_id = int(frame.text)
        first = int(start.text)
        frames[frame_id].append((first, first + int(duration.text)))
    phase_results = []
    for phase in phases:
        busy = []
        span = []
        for intervals in frames.values():
            first = min(item[0] for item in intervals)
            last = max(item[1] for item in intervals)
            center_unix = trace_start + (first + last) / 2e9
            if phase["start"] <= center_unix <= phase["end"]:
                busy.append(merged_duration(intervals) / 1e6)
                span.append((last - first) / 1e6)
        phase_results.append({"atlas": phase["atlas"],
                              "pcss_angular_deg": phase["pcss_angular_deg"],
                              "ssil_enabled": phase["ssil_enabled"], "gpu_frames": len(busy),
                              "gpu_busy": summarize(busy) if busy else None,
                              "gpu_span": summarize(span) if span else None})
    result = {"trace_start_unix_s": trace_start, "godot_gpu_frames": len(frames),
              "phases": phase_results,
              "note": "Metal interval busy time is an estimate of GPU work, not presented-frame latency."}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toc", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    run(args.toc, args.intervals, args.log)
