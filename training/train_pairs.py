"""Train a deterministic model on Godot shadow pairs; select by held-out scene."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.baseline import ResidualRenderer, select_device
from training.paired import ShadowPairs, evaluate


def build_model(kind: str):
    if kind == "cnn":
        return ResidualRenderer(max_delta=0.6)
    if kind == "multiscale":
        from models.multiscale import MultiScaleRenderer
        return MultiScaleRenderer(max_delta=0.6)
    if kind == "geometry":
        from models.multiscale import MultiScaleRenderer
        return MultiScaleRenderer(max_delta=0.6, condition_channels=11)
    if kind == "geometry_mask":
        from models.multiscale import MaskedShadowRenderer
        return MaskedShadowRenderer()
    if kind == "shadow_refine_mask":
        from models.multiscale import ShadowRefinementRenderer
        return ShadowRefinementRenderer()
    raise ValueError(kind)


def run(kind: str, epochs: int, device_name: str, root: Path, output: Path,
        seed: int = 7, changed_weight: float = 4.0, mask_pos_weight: float = 2.0):
    if changed_weight < 0 or mask_pos_weight <= 0:
        raise ValueError("loss weights must be valid")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    use_geometry = kind in {"geometry", "geometry_mask", "shadow_refine_mask"}
    train = ShadowPairs(root, "train", include_geometry=use_geometry)
    validation = ShadowPairs(root, "validation", include_geometry=use_geometry)
    test = ShadowPairs(root, "test", include_geometry=use_geometry)
    loader = DataLoader(train, batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = build_model(kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    baseline = {split: evaluate(None, data, device) for split, data in
                (("validation", validation), ("test", test))}
    best = float("inf")
    history = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if kind in {"geometry_mask", "shadow_refine_mask"}:
                prediction, mask_logits = model.forward_with_mask(x)
            else:
                prediction = model(x)
            changed = ((x[:, :3] - y).abs().amax(dim=1, keepdim=True) > 5 / 255).float()
            weights = 1 + changed_weight * changed
            loss = (weights * (prediction - y).abs()).mean()
            if kind in {"geometry_mask", "shadow_refine_mask"}:
                difference = x[:, :3] - y
                shadow_target = ((difference if kind == "geometry_mask" else difference.abs())
                                 .amax(dim=1, keepdim=True) > 5 / 255).float()
                loss = loss + 0.05 * F.binary_cross_entropy_with_logits(
                    mask_logits, shadow_target, pos_weight=torch.tensor(mask_pos_weight, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        val = evaluate(model, validation, device)
        history.append({"epoch": epoch + 1, "train_weighted_l1": float(np.mean(losses)),
                        "validation": val})
        if val["mae"] < best:
            best = val["mae"]
            torch.save({"model": model.state_dict(), "kind": kind, "epoch": epoch + 1,
                        "seed": seed}, output)
        print(f"epoch={epoch + 1} train={np.mean(losses):.5f} val_mae={val['mae']:.5f} best={best:.5f}", flush=True)
    state = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    result = {"kind": kind, "device": str(device), "epochs": epochs,
              "changed_weight": changed_weight,
              "mask_pos_weight": mask_pos_weight if kind in {"geometry_mask", "shadow_refine_mask"} else None,
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": state["epoch"], "input": baseline,
              "model": {"validation": evaluate(model, validation, device),
                        "test": evaluate(model, test, device)}, "history": history}
    (output.with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps({key: value for key, value in result.items() if key != "history"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("cnn", "multiscale", "geometry", "geometry_mask", "shadow_refine_mask"), default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shadow_cnn.pt"))
    parser.add_argument("--changed-weight", type=float, default=4.0)
    parser.add_argument("--mask-pos-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.changed_weight < 0 or args.mask_pos_weight <= 0:
        parser.error("invalid epochs or loss weights")
    run(args.kind, args.epochs, args.device, args.data, args.output,
        changed_weight=args.changed_weight, mask_pos_weight=args.mask_pos_weight)
