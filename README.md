# dlss66 — lightweight neural rendering research

Research and implementation of compact renderer-conditioned neural appearance models inspired by public DLSS 5 ideas. This project does not reproduce NVIDIA DLSS 5. The first implemented model is a **deterministic RGB residual CNN**, not diffusion. An original three-box Godot scene remains a rendering and capture testbed.

Current model docs: [published DLSS 5 architecture](docs/DLSS5_ARCHITECTURE.md), [our model design](docs/MODEL_DESIGN.md), [training method](docs/TRAINING_METHODOLOGY.md), and [executed experiments](docs/EXPERIMENT_LOG.md). Earlier [graphics research](docs/RESEARCH.md), [AO-centered architecture](docs/ARCHITECTURE.md), and [original plan](docs/IMPLEMENTATION_PLAN.md) remain for historical context; AO is an optional baseline.

## Minimal model

Requires Python 3, PyTorch, and NumPy. The synthetic pairs below are a training-pipeline smoke test only.

```sh
python3 -m unittest discover -s tests -v
python3 -m training.train --steps 8 --height 128 --width 128 --device auto
python3 -m evaluation.benchmark --device auto --height 144 --width 256
```

`auto` selects CUDA, then MPS, then CPU. Training writes an ignored checkpoint in `artifacts/`. See [MODEL_DESIGN.md](docs/MODEL_DESIGN.md) for exact tensor operations and [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for measured M2 results. Model-only timing excludes rendering and frame transfer.

## Godot capture

Download the [official Godot 4.7.2 macOS editor](https://github.com/godotengine/godot-builds/releases/tag/4.7.2-stable), unpack `Godot.app` into `tools/`, then run:

```sh
sh capture.sh
```

Expected output: `captures/scene.png`, `depth.png`, `normal.png`, and numerical `.npy` arrays. The command imports the compute shader, exports the renderer buffers, then checks their contents. See [TEST_RESULTS.md](TEST_RESULTS.md) for results and limits.

The official Godot binary is kept in `tools/` and ignored by Git. The test scene uses procedural boxes and no copyrighted game assets.
