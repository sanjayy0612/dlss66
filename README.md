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

This generates 480 local RGB pairs at 256×144 in ignored `dataset/raw/`, plus depth/normal geometry for each cheap frame, `manifest.jsonl`, and `summary.json`. Each pair uses the same procedural scene and camera; the cheap pass has directional shadows off and the reference pass has them on. The split is 336 train, 24 validation, 24 development, and 96 fresh test frames by scene. See [DATASET.md](docs/DATASET.md) for the exact format, validation results, and limitations. This is a small supervised shadow reconstruction dataset, not a DLSS 5 dataset.

The current, narrower shadow-refinement task preserves low-resolution shadows in the cheap input and uses a higher-resolution shadow atlas as the reference:

```sh
DLSS66_OUTPUT=dataset/shadow_resolution DLSS66_TASK=shadow_resolution DLSS66_LAYOUTS=32 sh collect_dataset.sh
python3 -m training.train_pairs --kind shadow_refine_mask --changed-weight 20 --epochs 20 --data dataset/shadow_resolution --device auto --output artifacts/shadow_refine_mask_32.pt
python3 -m evaluation.final_holdout --data dataset/shadow_resolution --kind shadow_refine_mask --checkpoint artifacts/shadow_refine_mask_32.pt --output artifacts/shadow_refine_holdout_32.json --device auto
python3 -m evaluation.calibrate_mask --kind shadow_refine_mask --data dataset/shadow_resolution --checkpoint artifacts/shadow_refine_mask_32.pt --device auto
DLSS66_SCENE_START=32 DLSS66_LAYOUTS=4 DLSS66_OUTPUT=dataset/shadow_resolution_fresh DLSS66_TASK=shadow_resolution sh collect_dataset.sh
python3 -m evaluation.final_holdout --data dataset/shadow_resolution_fresh --kind shadow_refine_mask --mask-bias 1.5 --checkpoint artifacts/shadow_refine_mask_32.pt --output artifacts/shadow_refine_holdout_fresh.json --device auto
python3 -m evaluation.temporal_blend --device auto --mask-bias 1.5
python3 -m evaluation.benchmark_refinement --device auto
python3 -m evaluation.visualize_refinement --data dataset/shadow_resolution_fresh --scenes 32 33 34 35 --mask-bias 1.5 --device auto
```

On the M2, the 80,152-parameter model reduced uncalibrated test MAE from 0.01136 to 0.01029 but initially worsened one scene. The current best uses `changed_weight=5`, `unchanged_weight=2`, and a hard confidence threshold selected on validation. At the stricter setting, it reduced MAE from 0.00987 to 0.00839 on four unseen static scenes and from 0.01266 to 0.01043 on two moving-sphere scenes. False darkening was 0.033% and 0.051%. Temporal correction lowered moving-sequence error from 0.00674 to 0.00634; cheap rendering was 0.00619. False darkening rose to 0.247% after blending. See [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## Live M2 feasibility check

The Godot/PyTorch bridge displays model output in a separate overlay. The latest aligned 48-frame run used **RGB-only conditioning**. The server uses the RGB checkpoint and calibrated hard gate:

```sh
python3 -m evaluation.serve_live --checkpoint artifacts/shadow_refine_rgb_balanced2_32.pt --kind shadow_refine_rgb --device mps --frames 48 --warmup 10 --mask-bias 4 --hard-mask
DLSS66_VARIANT=cheap DLSS66_TASK=shadow_resolution DLSS66_LIVE_RGB_ONLY=1 DLSS66_LIVE_FRAMES=48 DLSS66_OUTPUT=dataset/shadow_resolution tools/Godot.app/Contents/MacOS/Godot --path . -- --live
```

To run the full geometry-conditioned model instead, start the matching server and omit `DLSS66_LIVE_RGB_ONLY`:

```sh
python3 -m evaluation.serve_live --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --kind shadow_refine_mask --device mps --port 8766 --frames 48 --warmup 10 --mask-bias 4 --hard-mask
DLSS66_VARIANT=cheap DLSS66_TASK=shadow_resolution DLSS66_LIVE_FRAMES=48 DLSS66_OUTPUT=dataset/shadow_resolution tools/Godot.app/Contents/MacOS/Godot --path . -- --live
```

After startup, serial RGB-only processing took **27.70 ms median / 35.50 ms p95** at 256×144. The 11-channel geometry-conditioned model also completed a 48-frame run: **166.58 ms median / 167.22 ms p95**, about 6 FPS. It lowered mean RGB MAE from 0.005956 to 0.005544 on this repeated scene-28 integration sequence, with improvement in 32/48 frames. This is a small visual gain at an unacceptable cost for real-time use. The largest measured components were camera/render synchronization (66.60 ms), geometry export/readback wait (33.41 ms), RGB readback (18.15 ms), and presentation plus remaining work. These are host-observed stage times, not isolated GPU kernel durations. See [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for measurement details and limitations.

## Shadow-model experiments

With the local paired data, run the three experiments in order:

```sh
python3 -m training.train_pairs --kind cnn --epochs 20 --device auto --output artifacts/shadow_cnn.pt
python3 -m training.train_pairs --kind multiscale --epochs 20 --device auto --output artifacts/shadow_multiscale.pt
python3 -m training.train_one_step --teacher-steps 200 --student-steps 100 --device auto
python3 -m evaluation.compare_shadow --device auto
```

The third command trains a conditional diffusion noise predictor, samples it with eight deterministic DDIM steps, then distills those outputs into a one-call student. This is a small **diffusion-distillation experiment**, not NVIDIA's disclosed implementation. The original 96-pair results, where none beat the cheap input, are in [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) and can be reproduced from commit `778cc70`. These commands use the expanded dataset when run on current `main`. Checkpoints and comparison images are local in ignored `artifacts/`.

The newer geometry-conditioned comparison uses the expanded dataset:

```sh
python3 -m training.train_pairs --kind multiscale --epochs 20 --device auto --output artifacts/shadow_rgb_16.pt
python3 -m training.train_pairs --kind geometry --epochs 20 --device auto --output artifacts/shadow_geometry_16.pt
python3 -m evaluation.compare_geometry --device auto
```

The geometry model takes RGB, depth, view normals, a valid-depth mask, and view-space light direction. It improved aggregate held-out image error, but did not improve every scene. Its checkpoint and comparison image remain local in `artifacts/`. This older, harder shadow-creation task remains a useful failure case.
