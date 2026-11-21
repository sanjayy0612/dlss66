"""Export a spatial shadow-refinement checkpoint for a Godot GPU runtime.

The binary contains little-endian float32 tensors in PyTorch OIHW layout.
The JSON manifest records byte offsets, shapes, and the runtime thresholds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.multiscale import ShadowRefinementRenderer
from training.paired import ShadowPairs


def export(checkpoint: Path, output_dir: Path, mask_bias: float,
           data_root: Path, probe_scene: int, probe_frame: int):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("kind") != "shadow_refine_mask":
        raise ValueError("Godot exporter currently accepts spatial shadow_refine_mask checkpoints")
    model = state["model"]
    tensors = []
    chunks = []
    offset = 0
    for name, tensor in model.items():
        if not (name.endswith(".weight") or name.endswith(".bias")):
            raise ValueError(f"unexpected model tensor: {name}")
        value = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        payload = value.tobytes(order="C")
        tensors.append({"name": name, "shape": list(value.shape),
                        "offset_bytes": offset, "size_bytes": len(payload),
                        "layout": "OIHW" if value.ndim == 4 else "O"})
        chunks.append(payload)
        offset += len(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "weights.bin"
    manifest_path = output_dir / "manifest.json"
    binary_path.write_bytes(b"".join(chunks))

    dataset = ShadowPairs(data_root, "test", include_geometry=True)
    probe_index = next((index for index, row in enumerate(dataset.rows)
                        if row["scene_id"] == probe_scene and row["frame_id"] == probe_frame), None)
    if probe_index is None:
        raise ValueError(f"probe scene/frame not found: {probe_scene}/{probe_frame}")
    condition, _ = dataset[probe_index]
    conv_weight = model["enc0.0.weight"]
    conv_bias = model["enc0.0.bias"]
    runtime_model = ShadowRefinementRenderer().eval()
    runtime_model.load_state_dict(model)
    runtime_model.mask_bias = mask_bias
    runtime_model.hard_mask = True
    padded_condition = torch.zeros(12, *condition.shape[-2:], dtype=condition.dtype)
    padded_condition[:condition.shape[0]] = condition
    with torch.inference_mode():
        expected = F.conv2d(condition.unsqueeze(0), conv_weight, conv_bias, padding=1)[0]
        expected_model = runtime_model(condition.unsqueeze(0))[0]
    input_path = output_dir / "probe_input.bin"
    expected_path = output_dir / "probe_enc0_conv.bin"
    expected_model_path = output_dir / "probe_model_output.bin"
    input_path.write_bytes(padded_condition.numpy().astype("<f4", copy=False).tobytes(order="C"))
    expected_path.write_bytes(expected.numpy().astype("<f4", copy=False).tobytes(order="C"))
    expected_model_path.write_bytes(expected_model.numpy().astype("<f4", copy=False).tobytes(order="C"))
    manifest = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "checkpoint_kind": state["kind"],
        "epoch": state.get("epoch"),
        "seed": state.get("seed"),
        "input_channels": 11,
        "feature_channels": [12, 16, 24, 32],
        "max_delta": 0.6,
        "mask_bias": mask_bias,
        "hard_mask": True,
        "parameter_count": sum(int(np.prod(t["shape"])) for t in tensors),
        "binary_file": binary_path.name,
        "binary_size_bytes": offset,
        "tensors": tensors,
        "validation_probe": {
            "scene_id": probe_scene,
            "frame_id": probe_frame,
            "input_file": input_path.name,
            "input_shape": list(condition.shape),
            "input_buffer_channels": 12,
            "expected_enc0_conv_file": expected_path.name,
            "expected_enc0_conv_shape": list(expected.shape),
            "expected_model_output_file": expected_model_path.name,
            "expected_model_output_shape": list(expected_model.shape),
            "method": "PyTorch float32 Conv2d, 11 to 12 channels, 3x3, padding 1, before ReLU",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "weights": str(binary_path),
                      "parameters": manifest["parameter_count"],
                      "weights_bytes": offset}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_mask_balanced2_32.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/godot_model"))
    parser.add_argument("--mask-bias", type=float, default=4.0)
    parser.add_argument("--data", type=Path, default=Path("dataset/expanded_motion_test"))
    parser.add_argument("--probe-scene", type=int, default=48)
    parser.add_argument("--probe-frame", type=int, default=0)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.mask_bias, args.data,
           args.probe_scene, args.probe_frame)
