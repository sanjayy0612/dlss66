"""Small conditional diffusion teacher and one-step distillation experiment."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.baseline import select_device
from models.one_step import ConditionalNet, cosine_alphas, ddim_sample
from training.paired import ShadowPairs, evaluate


class DeterministicGenerator:
    def __init__(self, net, alphas=None):
        self.net = net
        self.alphas = alphas
        self.index = 0

    def eval(self):
        self.net.eval()
        self.index = 0

    def __call__(self, rgb):
        generator = torch.Generator().manual_seed(9000 + self.index)
        self.index += 1
        noise = torch.randn(rgb.shape, generator=generator).to(rgb.device)
        residual = (0.6 * torch.tanh(self.net(noise, rgb)) if self.alphas is None
                    else ddim_sample(self.net, rgb, noise, self.alphas))
        return (rgb + residual).clamp(0, 1)


def run(teacher_steps: int, student_steps: int, device_name: str, data: Path,
        output: Path, seed: int = 7) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    train = ShadowPairs(data, "train")
    validation = ShadowPairs(data, "validation")
    test = ShadowPairs(data, "test")
    loader = DataLoader(train, batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    teacher = ConditionalNet(with_time=True).to(device)
    student = ConditionalNet(with_time=False).to(device)
    alphas = cosine_alphas(8, device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    iterator = iter(loader)
    for step in range(teacher_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        residual = y - x
        indices = torch.randint(len(alphas), (len(x),), device=device)
        alpha = alphas[indices, None, None, None]
        noise = torch.randn_like(residual)
        noisy = torch.sqrt(alpha) * residual + torch.sqrt(1 - alpha) * noise
        prediction = teacher(noisy, x, indices.float() / (len(alphas) - 1))
        loss = F.mse_loss(prediction, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    teacher_noise_mse_last = float(loss.detach().cpu())
    teacher.eval()
    teacher_validation = evaluate(DeterministicGenerator(teacher, alphas), validation, device)
    teacher_test = evaluate(DeterministicGenerator(teacher, alphas), test, device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    iterator = iter(loader)
    for step in range(student_steps):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        x = x.to(device)
        noise = torch.randn_like(x)
        with torch.no_grad():
            teacher_residual = ddim_sample(teacher, x, noise, alphas)
        student_residual = 0.6 * torch.tanh(student(noise, x))
        loss = F.mse_loss(student_residual, teacher_residual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    student.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(),
                "seed": seed, "diffusion_steps": len(alphas)}, output)
    result = {"device": str(device), "teacher_steps": teacher_steps,
              "student_steps": student_steps, "diffusion_steps": len(alphas),
              "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
              "student_parameters": sum(p.numel() for p in student.parameters()),
              "teacher_noise_mse_last": teacher_noise_mse_last,
              "input": {"validation": evaluate(None, validation, device),
                        "test": evaluate(None, test, device)},
              "teacher": {"validation": teacher_validation, "test": teacher_test},
              "student": {"validation": evaluate(DeterministicGenerator(student), validation, device),
                          "test": evaluate(DeterministicGenerator(student), test, device)}}
    (output.with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-steps", type=int, default=200)
    parser.add_argument("--student-steps", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shadow_one_step.pt"))
    args = parser.parse_args()
    if min(args.teacher_steps, args.student_steps) < 1:
        parser.error("training steps must be positive")
    run(args.teacher_steps, args.student_steps, args.device, args.data, args.output)
