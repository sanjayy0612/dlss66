"""Train a recurrent geometry-conditioned renderer on three-frame rollouts."""

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
from models.temporal import TemporalShadowRefinementRenderer, initialize_from_spatial
from training.paired import ShadowPairs
from training.train_temporal import temporal_residual_loss, weighted_l1


class TemporalTriplets(Dataset):
    """Three consecutive paired frames and grids mapping each frame to its prior."""

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.frames = ShadowPairs(self.root, split, include_geometry=True)
        by_scene = defaultdict(list)
        for index, row in enumerate(self.frames.rows):
            by_scene[row["scene_id"]].append((row["frame_id"], index, row))
        self.triplets = []
        for sequence in by_scene.values():
            sequence.sort(key=lambda item: item[0])
            transitions = {}
            for previous, current in zip(sequence, sequence[1:]):
                if current[0] != previous[0] + 1:
                    continue
                # A current-frame pixel samples its corresponding previous-frame pixel.
                grid, visible = reprojection(self.root, current[2], previous[2])
                transitions[(previous[1], current[1])] = (grid[0], visible.unsqueeze(0).float())
            for first, second, third in zip(sequence, sequence[1:], sequence[2:]):
                if second[0] != first[0] + 1 or third[0] != second[0] + 1:
                    continue
                first_to_second = transitions.get((first[1], second[1]))
                second_to_third = transitions.get((second[1], third[1]))
                if first_to_second is None or second_to_third is None:
                    continue
                self.triplets.append((first[1], second[1], third[1],
                                      first_to_second, second_to_third))
        if not self.triplets:
            raise ValueError(f"no consecutive three-frame sequences in {root} ({split})")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        first_index, second_index, third_index, first_transition, second_transition = self.triplets[index]
        x0, y0 = self.frames[first_index]
        x1, y1 = self.frames[second_index]
        x2, y2 = self.frames[third_index]
        grid10, visible10 = first_transition
        grid21, visible21 = second_transition
        return x0, y0, x1, y1, x2, y2, grid10, visible10, grid21, visible21


def temporal_input(previous_prediction, previous_condition, grid, visible):
    """Warp a detached prior correction to the current image coordinates."""
    correction = (previous_prediction - previous_condition[:, :3]).detach()
    warped = F.grid_sample(correction, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False)
    valid = visible.to(dtype=previous_prediction.dtype)
    return warped * valid, valid


def blend_history_correction(prediction, current_rgb, history, visible, alpha):
    """Blend a prior reprojection into the current correction on visible pixels."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("history blend alpha must be between zero and one")
    correction = prediction - current_rgb
    candidate = torch.clamp(current_rgb + (1 - alpha) * correction + alpha * history,
                            0.0, 1.0)
    return torch.where(visible.bool(), candidate, prediction)


def supervised_frame_loss(prediction, logits, condition, reference,
                          changed_weight, unchanged_weight):
    rgb = condition[:, :3]
    supervised = weighted_l1(prediction, reference, rgb,
                             changed_weight, unchanged_weight)
    target_mask = ((rgb - reference).abs().amax(dim=1, keepdim=True) > (5 / 255)).float()
    mask = F.binary_cross_entropy_with_logits(
        logits, target_mask, pos_weight=torch.tensor(2.0, device=logits.device))
    return supervised, mask


@torch.inference_mode()
def evaluate_rollouts(model, data, device, mask_bias=4.0, hard_mask=True,
                      max_scenes=None, history_blend_alpha=0.0):
    """Run whole scene sequences with model predictions fed back as history."""
    from collections import defaultdict

    model.eval()
    model.mask_bias = mask_bias
    model.hard_mask = hard_mask
    grouped = defaultdict(list)
    for row, pair in zip(data.rows, data):
        grouped[row["scene_id"]].append((row, pair[0], pair[1]))
    scene_results = {}
    frame_maes = []
    input_maes = []
    temporal_errors = []
    cheap_temporal_errors = []
    unchanged_pixels = false_darkens = 0
    for scene_id, frames in grouped.items():
        if max_scenes is not None and len(scene_results) >= max_scenes:
            break
        frames.sort(key=lambda item: item[0]["frame_id"])
        predictions = []
        frame_scores = []
        for index, (row, condition, reference) in enumerate(frames):
            condition_device = condition.unsqueeze(0).to(device)
            if index == 0:
                history = torch.zeros_like(condition_device[:, :3])
                valid = torch.zeros_like(condition_device[:, :1])
            else:
                previous_row, previous_condition, _ = frames[index - 1]
                grid, visible = reprojection(data.root, row, previous_row)
                grid = grid.to(device)
                valid = visible[None, None].to(device=device, dtype=condition_device.dtype)
                previous_prediction = predictions[-1][None].to(device)
                history = F.grid_sample(
                    previous_prediction - previous_condition[:3][None].to(device),
                    grid, mode="bilinear", padding_mode="zeros", align_corners=False) * valid
            prediction, _ = model.forward_temporal(condition_device, history, valid)
            if history_blend_alpha > 0:
                prediction = blend_history_correction(
                    prediction, condition_device[:, :3], history, valid,
                    history_blend_alpha)
            prediction_cpu = prediction[0].cpu()
            predictions.append(prediction_cpu)
            mae = (prediction_cpu - reference).abs().mean().item()
            frame_maes.append(mae)
            input_maes.append((condition[:3] - reference).abs().mean().item())
            frame_scores.append(mae)
            untouched = (condition[:3] - reference).abs().amax(dim=0) <= (5 / 255)
            correction = prediction_cpu - condition[:3]
            unchanged_pixels += untouched.sum().item()
            false_darkens += (((-correction).amax(dim=0) > 5 / 255) & untouched).sum().item()
        for index in range(len(frames) - 1):
            row, _, reference = frames[index]
            next_row, _, next_reference = frames[index + 1]
            grid, visible = reprojection(data.root, row, next_row)
            warped_next = F.grid_sample(predictions[index + 1][None] - next_reference[None],
                                        grid, mode="bilinear", padding_mode="zeros",
                                        align_corners=False)[0]
            if visible.any():
                temporal_errors.append((predictions[index] - reference - warped_next)
                                       .abs()[:, visible].mean().item())
                current_cheap_error = frames[index][1][:3] - reference
                next_cheap_error = frames[index + 1][1][:3] - next_reference
                warped_cheap = F.grid_sample(next_cheap_error[None], grid, mode="bilinear",
                                             padding_mode="zeros", align_corners=False)[0]
                cheap_temporal_errors.append((current_cheap_error - warped_cheap)
                                             .abs()[:, visible].mean().item())
        scene_results[str(scene_id)] = {"mae": float(np.mean(frame_scores)),
                                        "frames": len(frame_scores)}
    return {"mae": float(np.mean(frame_maes)),
            "input_mae": float(np.mean(input_maes)),
            "temporal_error": float(np.mean(temporal_errors)) if temporal_errors else None,
            "input_temporal_error": float(np.mean(cheap_temporal_errors)) if cheap_temporal_errors else None,
            "unchanged_darken_rate_gt_5_255": false_darkens / max(unchanged_pixels, 1),
            "frames": len(frame_maes), "scenes": scene_results}


def run(data_root: Path, motion_data: Path, validation_data: Path,
        init_checkpoint: Path, output: Path, device_name="mps", epochs=5,
        batch_size=1, learning_rate=1e-4, temporal_weight=0.5,
        changed_weight=5.0, unchanged_weight=2.0, seed=7,
        max_train_triplets=None, max_validation_scenes=None):
    if epochs < 1 or batch_size < 1 or temporal_weight < 0:
        raise ValueError("epochs and batch size must be positive and temporal weight non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    train_sets = [TemporalTriplets(data_root, "train"),
                  TemporalTriplets(motion_data, "train")]
    train = ConcatDataset(train_sets)
    if max_train_triplets is not None:
        train = torch.utils.data.Subset(train, range(min(max_train_triplets, len(train))))
    static_validation = ShadowPairs(data_root, "validation", include_geometry=True)
    motion_validation = ShadowPairs(validation_data, "validation", include_geometry=True)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = TemporalShadowRefinementRenderer().to(device)
    initial = torch.load(init_checkpoint, map_location=device, weights_only=True)
    initialize_from_spatial(model, initial["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_objective = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(epochs):
        model.train()
        model.mask_bias = 0.0
        model.hard_mask = False
        losses = []
        for step, batch in enumerate(loader):
            if max_train_triplets is not None and step >= max_train_triplets:
                break
            (x0, y0, x1, y1, x2, y2, grid10, visible10,
             grid21, visible21) = batch
            x0, y0, x1, y1, x2, y2 = (value.to(device) for value in (x0, y0, x1, y1, x2, y2))
            grid10, visible10 = grid10.to(device), visible10.to(device)
            grid21, visible21 = grid21.to(device), visible21.to(device)
            zeros = torch.zeros_like(x0[:, :3])
            no_history = torch.zeros_like(x0[:, :1])
            p0, m0 = model.forward_temporal(x0, zeros, no_history)
            h1, v1 = temporal_input(p0, x0, grid10, visible10)
            p1, m1 = model.forward_temporal(x1, h1, v1)
            h2, v2 = temporal_input(p1, x1, grid21, visible21)
            p2, m2 = model.forward_temporal(x2, h2, v2)
            frame_losses = [supervised_frame_loss(p, mask, x, y,
                                                   changed_weight, unchanged_weight)
                            for p, mask, x, y in ((p0, m0, x0, y0),
                                                  (p1, m1, x1, y1), (p2, m2, x2, y2))]
            supervised = torch.stack([item[0] for item in frame_losses]).mean()
            mask_loss = torch.stack([item[1] for item in frame_losses]).mean()
            temporal = (temporal_residual_loss(p1, y1, p0, y0, grid10, visible10) +
                        temporal_residual_loss(p2, y2, p1, y1, grid21, visible21)) / 2
            loss = supervised + 0.05 * mask_loss + temporal_weight * temporal
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((loss.detach().item(), supervised.detach().item(), temporal.detach().item()))
        model.eval()
        static_metrics = evaluate_rollouts(model, static_validation, device,
                                           max_scenes=max_validation_scenes)
        motion_metrics = evaluate_rollouts(model, motion_validation, device,
                                           max_scenes=max_validation_scenes)
        objective = (static_metrics["mae"] + motion_metrics["mae"]) / 2 + temporal_weight * motion_metrics["temporal_error"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean([x[0] for x in losses])),
                        "train_supervised_l1": float(np.mean([x[1] for x in losses])),
                        "train_temporal_l1": float(np.mean([x[2] for x in losses])),
                        "validation_static": static_metrics, "validation_motion": motion_metrics,
                        "validation_objective": objective})
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "kind": "shadow_refine_temporal",
                        "epoch": best_epoch, "seed": seed,
                        "temporal_weight": temporal_weight,
                        "init_checkpoint": str(init_checkpoint)}, output)
        print(f"epoch={epoch + 1} train={history[-1]['train_loss']:.6f} "
              f"val_static={static_metrics['mae']:.6f} "
              f"val_motion={motion_metrics['mae']:.6f} "
              f"val_temporal={motion_metrics['temporal_error']:.6f} "
              f"best={best_objective:.6f}", flush=True)
    result = {"device": str(device), "data": str(data_root), "motion_data": str(motion_data),
              "validation_data": str(validation_data), "init_checkpoint": str(init_checkpoint),
              "output": str(output), "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "temporal_weight": temporal_weight,
              "changed_weight": changed_weight, "unchanged_weight": unchanged_weight,
              "train_triplets": len(train), "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": best_epoch, "best_validation_objective": best_objective,
              "history": history}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/shadow_resolution"))
    parser.add_argument("--motion-data", type=Path, default=Path("dataset/motion_train"))
    parser.add_argument("--validation-data", type=Path, default=Path("dataset/motion_validation"))
    parser.add_argument("--init-checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_mask_balanced2_32.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/shadow_refine_recurrent_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--changed-weight", type=float, default=5.0)
    parser.add_argument("--unchanged-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-triplets", type=int, default=None)
    parser.add_argument("--max-validation-scenes", type=int, default=None)
    args = parser.parse_args()
    run(args.data, args.motion_data, args.validation_data, args.init_checkpoint,
        args.output, args.device, args.epochs, args.batch_size,
        args.learning_rate, args.temporal_weight, args.changed_weight,
        args.unchanged_weight, args.seed, args.max_train_triplets,
        args.max_validation_scenes)
