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
    raise ValueError(kind)


def run(kind: str, epochs: int, device_name: str, root: Path, output: Path, seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    train = ShadowPairs(root, "train")
    validation = ShadowPairs(root, "validation")
    test = ShadowPairs(root, "test")
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
            prediction = model(x)
            changed = ((x - y).abs().amax(dim=1, keepdim=True) > 5 / 255).float()
            weights = 1 + 4 * changed
            loss = (weights * (prediction - y).abs()).mean()
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
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": state["epoch"], "input": baseline,
              "model": {"validation": evaluate(model, validation, device),
                        "test": evaluate(model, test, device)}, "history": history}
    (output.with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps({key: value for key, value in result.items() if key != "history"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("cnn", "multiscale"), default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shadow_cnn.pt"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.kind, args.epochs, args.device, args.data, args.output)
