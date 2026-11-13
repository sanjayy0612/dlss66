# dlss66 — lightweight neural rendering research

Research and implementation of compact renderer-conditioned neural appearance models inspired by public DLSS 5 ideas. This project does not reproduce NVIDIA DLSS 5. It now compares a **deterministic RGB residual CNN**, a small multiscale model, and a one-call student distilled from a conditional diffusion teacher. An original three-box Godot scene remains a rendering and capture testbed.

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

## Paired renderer data

With the same pinned Godot app and Python packages NumPy and Pillow, run:

```sh
sh collect_dataset.sh
```

This generates 96 local pairs at 256×144 in ignored `dataset/raw/`, plus `manifest.jsonl` and `summary.json`. Each pair uses the same procedural scene and camera; the cheap pass has directional shadows off and the reference pass has them on. Scene IDs 0–5 are training, 6 is validation, and 7 is test. See [DATASET.md](docs/DATASET.md) for the exact task, validation results, and limitations. This is a small supervised shadow reconstruction dataset, not a DLSS 5 dataset.

## Shadow-model experiments

With the local paired data, run the three experiments in order:

```sh
python3 -m training.train_pairs --kind cnn --epochs 20 --device auto --output artifacts/shadow_cnn.pt
python3 -m training.train_pairs --kind multiscale --epochs 20 --device auto --output artifacts/shadow_multiscale.pt
python3 -m training.train_one_step --teacher-steps 200 --student-steps 100 --device auto
python3 -m evaluation.compare_shadow --device auto
```

The third command trains a conditional diffusion noise predictor, samples it with eight deterministic DDIM steps, then distills those outputs into a one-call student. This is a small **diffusion-distillation experiment**, not NVIDIA's disclosed implementation. Current M2 results show that none of the three trained models beats the cheap input on held-out image MAE; see [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md). Checkpoints and comparison images are local in ignored `artifacts/`.
