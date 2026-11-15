"""Single-client, fixed-size Godot RGB/geometry to MPS inference bridge.

Protocol per frame: RGB8 HWC, RGBA32F geometry, three float32 light values;
response: RGB8 HWC. All sizes are fixed at 256x144 for this experiment.
"""

import argparse
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.benchmark import synchronize
from models.baseline import select_device
from training.train_pairs import build_model

HEIGHT, WIDTH = 144, 256
RGB_BYTES = HEIGHT * WIDTH * 3
GEOMETRY_BYTES = HEIGHT * WIDTH * 16


def receive_exact(connection, length):
    chunks = []
    left = length
    while left:
        data = connection.recv(left)
        if not data:
            raise ConnectionError(f"connection closed with {left} bytes pending")
        chunks.append(data)
        left -= len(data)
    return b"".join(chunks)


def condition_tensor(rgb_bytes, geometry_bytes, light_bytes):
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).astype(np.float32) / 255.0
    geometry = np.frombuffer(geometry_bytes, dtype="<f4").reshape(HEIGHT, WIDTH, 4)
    device_depth = geometry[:, :, 0]
    valid = (device_depth > 0).astype(np.float32)
    near, far = 0.05, 100.0
    meters = np.zeros_like(device_depth)
    meters[valid > 0] = near * far / (near + device_depth[valid > 0] * (far - near))
    depth = np.clip(meters / 10.0, 0, 1)
    light = np.asarray(struct.unpack("<fff", light_bytes), dtype=np.float32)
    if not np.isfinite(geometry).all() or not np.isfinite(light).all():
        raise ValueError("non-finite renderer input")
    channels = np.concatenate((rgb, depth[:, :, None], geometry[:, :, 1:4],
                               valid[:, :, None], np.broadcast_to(light, (HEIGHT, WIDTH, 3))), axis=2)
    return torch.from_numpy(np.ascontiguousarray(channels)).permute(2, 0, 1).unsqueeze(0).contiguous()


def run(checkpoint: Path, device_name: str, port: int, frames: int, warmup: int):
    device = select_device(device_name)
    model = build_model("shadow_refine_mask").to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.mask_bias = 1.5
    references = []
    for frame_id in range(12):
        path = Path(f"dataset/shadow_resolution/scene_28/frame_{frame_id:03d}_reference.png")
        references.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) if path.exists() else None)
    with torch.inference_mode():
        dummy = torch.zeros(1, 11, HEIGHT, WIDTH, device=device)
        for _ in range(warmup):
            model(dummy)
        synchronize(device)
    measurements = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        listener.settimeout(60)
        print(f"LIVE_SERVER_READY port={port} device={device}", flush=True)
        connection, _ = listener.accept()
        with connection, torch.inference_mode():
            connection.settimeout(10)
            for frame in range(frames):
                start = time.perf_counter()
                rgb = receive_exact(connection, RGB_BYTES)
                geometry = receive_exact(connection, GEOMETRY_BYTES)
                light = receive_exact(connection, 12)
                received = time.perf_counter()
                condition = condition_tensor(rgb, geometry, light)
                prepared = time.perf_counter()
                prediction = model(condition.to(device))
                synchronize(device)
                inferred = time.perf_counter()
                output = (prediction[0].cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                packed = output.tobytes()
                if len(packed) != RGB_BYTES:
                    raise ValueError("wrong output size")
                connection.sendall(packed)
                sent = time.perf_counter()
                reference = references[frame % 12]
                cheap_mae = (float(np.mean(np.abs(np.frombuffer(rgb, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                                                .astype(np.float32) - reference)) / 255) if reference is not None else None)
                model_mae = (float(np.mean(np.abs(output.astype(np.float32) - reference)) / 255)
                             if reference is not None else None)
                measurements.append({"id": frame,
                       "receive_ms": (received - start) * 1000,
                       "prepare_ms": (prepared - received) * 1000,
                       "model_sync_ms": (inferred - prepared) * 1000,
                       "output_send_ms": (sent - inferred) * 1000,
                       "cheap_mae": cheap_mae, "model_mae": model_mae})
                print("LIVE_SERVER_FRAME", json.dumps(measurements[-1]), flush=True)
    output_path = Path("artifacts/live_server.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_refine_mask_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.port, args.frames, args.warmup)
