"""Fine-tune a shadow refinement CNN with reprojection-based temporal loss."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from evaluation.temporal import reprojection
from models.baseline import select_device
from training.paired import ShadowPairs, evaluate
from training.train_pairs import build_model


class TemporalPairDataset(Dataset):
    """Consecutive paired frames with a depth-tested source-to-target grid."""

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.frames = ShadowPairs(self.root, split, include_geometry=True)
        by_scene = defaultdict(list)
        for index, row in enumerate(self.frames.rows):
            by_scene[row["scene_id"]].append((row["frame_id"], index, row))
        self.pairs = []
        for sequence in by_scene.values():
            sequence.sort(key=lambda item: item[0])
            for source, target in zip(sequence, sequence[1:]):
                if target[0] != source[0] + 1:
                    continue
                grid, visible = reprojection(self.root, source[2], target[2])
                self.pairs.append((source[1], target[1], grid[0], visible.unsqueeze(0).float()))
        if not self.pairs:
            raise ValueError(f"no consecutive temporal pairs in {self.root} ({split})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source_index, target_index, grid, visible = self.pairs[index]
        source_condition, source_reference = self.frames[source_index]
        target_condition, target_reference = self.frames[target_index]
        return (source_condition, source_reference, target_condition, target_reference,
                grid, visible)


def weighted_l1(prediction, target, input_rgb, changed_weight=5.0, unchanged_weight=2.0):
    changed = (input_rgb - target).abs().amax(dim=1, keepdim=True) > (5 / 255)
    weights = changed.float() * (1 + changed_weight) + (~changed).float() * unchanged_weight
    return (weights * (prediction - target).abs()).mean()


def temporal_residual_terms(source_prediction, source_reference,
                            target_prediction, target_reference, grid, visible):
    source_error = source_prediction - source_reference
    target_error = target_prediction - target_reference
    warped_target_error = F.grid_sample(target_error, grid, mode="bilinear",
                                        padding_mode="zeros", align_corners=False)
    valid = visible.expand_as(source_error) > 0
    numerator = ((source_error - warped_target_error).abs() * valid).sum()
    return numerator, valid.sum()


def temporal_residual_loss(source_prediction, source_reference,
                           target_prediction, target_reference, grid, visible):
    numerator, count = temporal_residual_terms(source_prediction, source_reference,
                                               target_prediction, target_reference,
                                               grid, visible)
    return numerator / count.clamp_min(1)


@torch.inference_mode()
def evaluate_temporal(model, dataset, device, batch_size):
    model.eval()
    numerator = count = 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x0, y0, x1, y1, grid, visible in loader:
        x0, y0, x1, y1 = (value.to(device) for value in (x0, y0, x1, y1))
        grid, visible = grid.to(device), visible.to(device)
        prediction0 = model(x0)
        prediction1 = model(x1)
        batch_sum, batch_count = temporal_residual_terms(
            prediction0, y0, prediction1, y1, grid, visible)
        numerator += batch_sum.item()
        count += batch_count.item()
    return numerator / max(count, 1.0)


def run(data_root: Path, init_checkpoint: Path, output: Path, device_name="mps",
        epochs=5, batch_size=2, learning_rate=1e-4, temporal_weight=0.5,
        changed_weight=5.0, unchanged_weight=2.0, seed=7,
        motion_data: Path | None = None, validation_data: Path | None = None):
    if epochs < 1 or batch_size < 1 or temporal_weight < 0:
        raise ValueError("epochs and batch size must be positive and temporal weight non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    train_sets = [TemporalPairDataset(data_root, "train")]
    if motion_data is not None:
        train_sets.append(TemporalPairDataset(motion_data, "train"))
    train = ConcatDataset(train_sets)
    static_validation = ShadowPairs(data_root, "validation", include_geometry=True)
    validation_root = validation_data or data_root
    motion_validation = ShadowPairs(validation_root, "validation", include_geometry=True)
    temporal_validation = TemporalPairDataset(validation_root, "validation")
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = build_model("shadow_refine_mask").to(device)
    initial = torch.load(init_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(initial["model"])
    baseline_static = evaluate(model, static_validation, device)
    baseline_motion = evaluate(model, motion_validation, device)
    baseline_temporal = evaluate_temporal(model, temporal_validation, device, batch_size)
    baseline_objective = ((baseline_static["mae"] + baseline_motion["mae"]) / 2 +
                          temporal_weight * baseline_temporal)
    best_objective = baseline_objective
    best_epoch = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "kind": "shadow_refine_mask",
                "epoch": 0, "seed": seed, "temporal_weight": 0.0,
                "init_checkpoint": str(init_checkpoint)}, output)
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        totals = []
        for x0, y0, x1, y1, grid, visible in loader:
            x0, y0, x1, y1 = (value.to(device) for value in (x0, y0, x1, y1))
            grid, visible = grid.to(device), visible.to(device)
            p0, logits0 = model.forward_with_mask(x0)
            p1, logits1 = model.forward_with_mask(x1)
            supervised = (weighted_l1(p0, y0, x0[:, :3], changed_weight, unchanged_weight) +
                          weighted_l1(p1, y1, x1[:, :3], changed_weight, unchanged_weight)) / 2
            shadow0 = ((x0[:, :3] - y0).abs().amax(dim=1, keepdim=True) > (5 / 255)).float()
            shadow1 = ((x1[:, :3] - y1).abs().amax(dim=1, keepdim=True) > (5 / 255)).float()
            mask_loss = (F.binary_cross_entropy_with_logits(
                logits0, shadow0, pos_weight=torch.tensor(2.0, device=device)) +
                F.binary_cross_entropy_with_logits(
                    logits1, shadow1, pos_weight=torch.tensor(2.0, device=device))) / 2
            temporal = temporal_residual_loss(p0, y0, p1, y1, grid, visible)
            loss = supervised + 0.05 * mask_loss + temporal_weight * temporal
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append({"loss": loss.detach().item(),
                           "supervised": supervised.detach().item(),
                           "temporal": temporal.detach().item()})
        val_static = evaluate(model, static_validation, device)
        val_motion = evaluate(model, motion_validation, device)
        val_temporal = evaluate_temporal(model, temporal_validation, device, batch_size)
        val_objective = (val_static["mae"] + val_motion["mae"]) / 2 + temporal_weight * val_temporal
        history.append({"epoch": epoch + 1,
                        "train_loss": float(np.mean([row["loss"] for row in totals])),
                        "train_supervised_l1": float(np.mean([row["supervised"] for row in totals])),
                        "train_temporal_l1": float(np.mean([row["temporal"] for row in totals])),
                        "validation_static": val_static,
                        "validation_motion": val_motion,
                        "validation_temporal_error": val_temporal,
                        "validation_objective": val_objective})
        if val_objective < best_objective:
            best_objective = val_objective
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "kind": "shadow_refine_mask",
                        "epoch": best_epoch, "seed": seed,
                        "temporal_weight": temporal_weight,
                        "init_checkpoint": str(init_checkpoint)}, output)
        print(f"epoch={epoch + 1} train={history[-1]['train_loss']:.6f} "
              f"temporal={history[-1]['train_temporal_l1']:.6f} "
              f"val_motion_mae={val_motion['mae']:.6f} "
              f"val_temporal={val_temporal:.6f} objective={val_objective:.6f} "
              f"best={best_objective:.6f}", flush=True)
    state = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    result = {"device": str(device), "data": str(data_root),
              "motion_data": str(motion_data) if motion_data is not None else None,
              "validation_data": str(validation_root),
              "init_checkpoint": str(init_checkpoint), "output": str(output),
              "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "temporal_weight": temporal_weight,
              "changed_weight": changed_weight, "unchanged_weight": unchanged_weight,
              "train_pairs": len(train),
              "static_validation_pairs": len(static_validation),
              "motion_validation_pairs": len(motion_validation),
              "motion_validation_transitions": len(temporal_validation),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "best_epoch": best_epoch, "baseline_objective": baseline_objective,
              "baseline_static_validation": baseline_static,
              "baseline_motion_validation": baseline_motion,
              "baseline_temporal_validation": baseline_temporal,
              "model_static_validation": evaluate(model, static_validation, device),
              "model_motion_validation": evaluate(model, motion_validation, device),
              "model_temporal_validation": evaluate_temporal(model, temporal_validation,
                                                               device, batch_size),
              "history": history}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps({key: value for key, value in result.items() if key != "history"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/shadow_resolution"))
    parser.add_argument("--motion-data", type=Path, default=None,
                        help="Optional moving-object training sequences, kept separate from validation/test")
    parser.add_argument("--validation-data", type=Path, default=None,
                        help="Optional paired motion-validation root")
    parser.add_argument("--init-checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_mask_balanced2_32.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shadow_refine_temporal_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--changed-weight", type=float, default=5.0)
    parser.add_argument("--unchanged-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.data, args.init_checkpoint, args.output, args.device, args.epochs,
        args.batch_size, args.learning_rate, args.temporal_weight,
        args.changed_weight, args.unchanged_weight, args.seed,
        args.motion_data, args.validation_data)
