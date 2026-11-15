# Experiment log

## Executed 2026-11-12

Environment: MacBook Air M2, 8 GB unified memory; Python 3.13; PyTorch 2.13.0; native MPS. Existing Godot RGB/depth/normal capture remains available. The new model currently uses RGB only.

| Experiment | Command | Result |
|---|---|---|
| CPU shape, identity, gradient, synthetic determinism, checkpoint round-trip | `python3 -m unittest discover -s tests -v` | 3 tests passed in 0.981 s |
| Short MPS training smoke | `python3 -m training.train --steps 8 --height 128 --width 128 --device mps` | 10,163 parameters; initial train L1 0.09796; held-out input L1 0.09938; model L1 0.02902; ~45 KiB checkpoint saved |
| M2 model-only inference | `python3 -m evaluation.benchmark --device mps --height 144 --width 256 --warmup 10 --samples 30` | mean 1.520 ms; median 1.487 ms; p95 1.768 ms, with MPS synchronization each sample |

The synthetic target is a fixed lighting/color transform on procedurally generated rectangles. The training result proves the software path learns that simple mapping; it is **not evidence** of realistic appearance generation, scene transfer, temporal stability, or DLSS-like quality. The inference result excludes Godot rendering, frame readback, tensor creation, compositing, and presentation. No memory peak or full-frame FPS was measured. No generative model was trained.

Earlier renderer capture experiments and their limits are in [TEST_RESULTS.md](../TEST_RESULTS.md).

## Paired renderer capture, executed 2026-11-12

`sh collect_dataset.sh` ran Godot 4.7.2 on the M2 Metal Forward+ renderer. It captured 96 pairs of 256×144 RGB PNGs from eight procedural scene layouts and 12 camera poses per layout. The two passes differed by directional shadows off versus on. `validate_dataset.py` checked matching seed, camera pose, dimensions, file presence, unique keys, and nonzero image differences. The scene split is 72 train, 12 validation, and 12 test pairs. Mean absolute cheap/reference difference was 10.199/255 per channel; the smallest pair was 2.041/255 and the largest 18.297/255. On average, 12.31% of pixels differed by more than five levels in at least one channel. A four-pair pilot produced identical SHA-256 PNG hashes on two runs. One sampled PNG pair and the manifest had matching hashes across two full runs. Full-corpus hash reproducibility and geometry-mask alignment were not independently measured. This original data generator is preserved in commit `387b4fc`; the current [dataset](DATASET.md) is the expanded version.

## Planned, not executed

- Repeat full-corpus hashes and verify object-mask alignment if this dataset is used for a quantitative training claim.
- Expand scene and lighting diversity beyond the current 16 procedural layouts before interpreting shadow results as a general limit of neural rendering.
- Compare a useful model against a conventional shadow baseline at matched end-to-end cost.
- Evaluate temporal stability with moving objects and motion vectors. Fixed still-image seeds alone do not prevent video flicker.

## Shadow-model comparison, executed 2026-11-13

All models used the same 72 train, 12 validation, and 12 test Godot pairs at 256×144 on the MacBook Air M2 with PyTorch MPS. Scene 7 was held out from optimization and checkpoint selection. The deterministic models trained for 20 epochs with a `1+4×changed-pixel` weighted L1 objective; the best validation **whole-image MAE** checkpoint was selected. The CNN's configurable output bound was raised from 0.25 to 0.6 because 10.7% of target color-channel changes exceed 0.25. The diffusion teacher trained for 200 updates with DDPM noise prediction across eight noise levels. Its one-call student trained for 100 updates to match deterministic eight-step DDIM teacher samples. These short runs are architecture probes, not tuned training results.

| Test scene 7 | Parameters | MAE ↓ | PSNR dB ↑ | Changed-pixel MAE ↓ | Model-call mean ms ↓ |
|---|---:|---:|---:|---:|---:|
| Cheap input | — | **0.04396** | 18.28 | 0.31263 | — |
| Residual CNN, best epoch 1 | 10,163 | 0.05023 | 18.68 | **0.29599** | 1.52 |
| Multiscale CNN, best epoch 2 | 79,179 | 0.04400 | 18.28 | 0.31248 | 2.42 |
| Conditional diffusion teacher, eight DDIM calls | 30,979 | 0.23268 | 10.65 | 0.34123 | 13.78 |
| Distilled student, one call | 30,835 | 0.21383 | 11.43 | 0.33549 | 2.07 |

The CNN modestly reduced changed-pixel error but altered many unchanged pixels, so whole-image MAE became worse. The multiscale model learned almost identity. Both diffusion outputs contain severe noise; the student is faster and slightly less wrong than its teacher, but neither is a useful enhancement. The teacher's last training-batch noise MSE was 0.110; this alone did not predict sampled-image quality. No VAE was used. The comparison image is local at `artifacts/shadow_comparison/test_scene_7_frame_0.png` in order: input, reference, CNN, multiscale, teacher, student.

Commands executed:

```sh
python3 -m unittest discover -s tests -q
python3 -m training.train_pairs --kind cnn --epochs 20 --device mps --output artifacts/shadow_cnn.pt
python3 -m training.train_pairs --kind multiscale --epochs 20 --device mps --output artifacts/shadow_multiscale.pt
python3 -m training.train_one_step --teacher-steps 200 --student-steps 100 --device mps
python3 -m evaluation.compare_shadow --device mps
```

The timing uses 10 warmups and 30 synchronized MPS samples per model at 256×144. CNN p95 was 1.87 ms, multiscale 2.49 ms, teacher 14.48 ms, and student 2.42 ms. It excludes Godot rendering, readback, tensor creation, and presentation. Peak memory, SSIM, LPIPS, model-frame temporal consistency, and matched conventional shadow-rendering cost remain unmeasured. Checkpoints and full result JSON files are in ignored `artifacts/` and can be regenerated with the commands above. The one-step result establishes the code path and cost difference, **not** DLSS 5 quality or real-time integration.

## Geometry-guided capture and matched model comparison, executed 2026-11-13

`sh collect_dataset.sh` regenerated the dataset with 16 layouts × 12 camera poses using Godot 4.7.2, Metal Forward+ on the M2. The cheap pass exported 192 aligned RGBA32F depth/normal frames through `dataset_probe.gd`; the reference pass saved shadowed RGB from the same seeded geometry and camera. The split is 144 train, 24 validation, and 24 test pairs by scene. The validator found all 192 geometry files at exactly 589,824 bytes, finite reverse-Z depth, foreground unit normals (maximum length error 1.2×10⁻⁷), and zero mismatch between cheap/reference sky masks. View depth ranged 0.800–9.332 m. All geometry hashes were distinct. The local dataset is about 112 MiB and is not committed.

One early full capture stalled while waiting for `frame_post_draw`; removing that wait and forcing draws resolved the four-pair pilot. A later run timed out when the compositor was not being redrawn; requesting draws during the asynchronous readback wait allowed the full 192-pair run to complete. The timeout run emitted RID leak warnings on abrupt quit; the error path now calls compositor cleanup. The final capture completed without reported Godot errors. These are execution-path observations, not a measured GPU readback latency for the new dataset.

Two models used the same multiscale backbone, 20 epochs, weighted L1 loss, seed 7, and scene split. The RGB-only model had 79,179 parameters; the geometry-conditioned model had 80,043 and consumed 11 channels (RGB, metric depth, view normals, valid mask, view-space light direction). Checkpoints were selected by validation whole-image MAE before evaluating test scenes 14 and 15.

| 16-layout test set | MAE ↓ | PSNR dB ↑ | Changed-pixel MAE ↓ | Unchanged-pixel MAE ↓ |
|---|---:|---:|---:|---:|
| Cheap input | 0.03978 | 18.67 | 0.30747 | 0.00002 |
| RGB-only multiscale, best epoch 1 | 0.04247 | 18.77 | 0.30307 | 0.00375 |
| Geometry-conditioned, best epoch 18 | **0.03878** | **20.76** | **0.12387** | 0.02614 |

The aggregate gain hides uneven scene transfer. The geometry model improved 10 of 24 test frames. Scene 14 mean MAE worsened from 0.01026 (input) to 0.03401; scene 15 improved from 0.06930 to 0.04354. The saved comparison at `artifacts/geometry_comparison/test_scene_14_frame_0.png` shows dark halos around some objects. The model appears to learn some shadow placement but also changes unshadowed regions; it is not a robust renderer yet.

Commands executed:

```sh
sh collect_dataset.sh
python3 -m unittest discover -s tests -q
python3 -m training.train_pairs --kind multiscale --epochs 20 --device mps --output artifacts/shadow_rgb_16.pt
python3 -m training.train_pairs --kind geometry --epochs 20 --device mps --output artifacts/shadow_geometry_16.pt
python3 -m evaluation.compare_geometry --device mps
```

The first synchronized 30-sample model-only timing run at 256×144 measured mean 2.44 ms for RGB-only and 2.22 ms for geometry. A repeat measured means 3.12 and 2.91 ms, with p95 6.49 and 5.48 ms; the variation makes a speed ranking unjustified. These timings exclude Godot rendering, readback, tensor construction, and display. End-to-end FPS, peak memory, temporal stability, and a matched conventional shadow baseline remain unmeasured. The practical next experiment is to improve cross-scene coverage and penalize false shadow changes before returning to the one-step generator.

## Cross-scene check of shadow creation and refinement, executed 2026-11-13–14

We expanded the original shadow-off task to 40 scenes (480 frames). Scenes 0–11 and 16–31 trained the model, 12–13 selected checkpoints, 14–15 served as development, and fresh scenes 32–39 were held out. The masked 80,152-parameter model reached fresh-scene aggregate MAE **0.02742** versus **0.03371** for cheap RGB and PSNR **21.44** versus **19.30** dB. It improved only **4 of 8 scenes** and **59 of 96 frames**. The model still invented dark halos when shadows were weak. Reducing changed-pixel weight to 1 made a near-identity output; tuning the mask positive weight did not solve the held-out failure. The full local per-scene report is `artifacts/final_holdout_40.json`. This failure motivated a better specified task with an actual cheap shadow cue.

The new `shadow_resolution` task has directional shadows in both input and reference. Godot 4.7.2 sets the cheap directional shadow atlas to 512 and the reference atlas to 4096 using `RenderingServer.directional_shadow_atlas_set_size`. The M2 capture completed for 32 scenes × 12 views = **384** aligned 256×144 RGB and geometry pairs. The scene split is 288 train, 48 validation, 48 test pairs. The validator found mean cheap/reference absolute difference **2.620/255**, 9.17% pixels changed by more than five levels, finite reverse-Z depth, unit normals with maximum length error 1.2×10⁻⁷, 0.765–9.332 m foreground depth, and zero sky-mask mismatch. The local dataset uses about 224 MiB. The 128-pixel atlas pilot had conspicuous striped artifacts, so 512 was selected for the full task; that selection used pilot views, not the test split. The reference is a conventional renderer choice, not physical truth.

All models used the same 11 conditioning channels: cheap RGB, depth, normals, geometry-valid mask, and view-space light direction. A 20-epoch unmasked multiscale model with changed-pixel weight 4 stayed near identity. Raising that weight to 20 improved test PSNR but worsened MAE by altering unmodified regions. The selected model added a learned change mask and signed, mask-gated RGB correction. It trained for 20 epochs with seed 7, weighted L1 (`1+20×` on pixels changed by the reference), and a small supervised mask BCE term; the best checkpoint was chosen by validation MAE (epoch 20). Signed correction is necessary because the 512-to-4096 reference difference contains both darker and brighter pixels. The model has 80,152 parameters.

| Fresh test, scenes 28–31 | Cheap 512 atlas | Masked neural refinement | High-quality 4096 atlas |
|---|---:|---:|---:|
| Whole-image MAE ↓ | 0.011364 | **0.010289** | 0 by definition |
| PSNR dB ↑ | 27.344 | **29.954** | Reference |
| Changed-pixel MAE ↓ | 0.103711 | **0.056796** | 0 |
| Unchanged-pixel MAE ↓ | **0.000645** | 0.004890 | 0 |
| Improved frames | — | 33 / 48 | — |
| Improved scenes | — | 3 / 4 | — |

Scene-level MAE: scene 28 **worsened** 0.005956→0.007295 (0/12 frames improved); scene 29 improved 0.010715→0.007891 (12/12); scene 30 improved 0.013346→0.011659 (12/12); scene 31 improved 0.015437→0.014309 (9/12). Visual inspection of `artifacts/shadow_refine_comparison.png` confirmed false darkening on the floor in scene 28. The aggregate gain is therefore **not robust cross-scene generalization** and does not establish a useful net performance gain. Repeated test inspection makes scenes 28–31 diagnostic data for future revisions; use new unseen scenes for final claims.

Commands executed:

```sh
DLSS66_OUTPUT=dataset/shadow_resolution DLSS66_TASK=shadow_resolution DLSS66_LAYOUTS=32 sh collect_dataset.sh
DLSS66_OUTPUT=dataset/shadow_resolution python3 validate_dataset.py
python3 -m training.train_pairs --kind geometry --epochs 20 --data dataset/shadow_resolution --device mps --output artifacts/shadow_refine_geometry_32.pt
python3 -m training.train_pairs --kind geometry --changed-weight 20 --epochs 20 --data dataset/shadow_resolution --device mps --output artifacts/shadow_refine_geometry_32_weight20.pt
python3 -m training.train_pairs --kind shadow_refine_mask --changed-weight 20 --epochs 20 --data dataset/shadow_resolution --device mps --output artifacts/shadow_refine_mask_32.pt
python3 -m evaluation.final_holdout --data dataset/shadow_resolution --kind shadow_refine_mask --checkpoint artifacts/shadow_refine_mask_32.pt --output artifacts/shadow_refine_holdout_32.json --device mps
python3 -m evaluation.visualize_refinement --device mps
python3 -m evaluation.benchmark_refinement --device mps --warmup 10 --samples 100
python3 -m unittest discover -s tests -q
```

The final unit run passed **9 tests**. The synchronized M2 benchmark at 256×144 measured model-only mean **2.854 ms**, median **2.303 ms**, p95 **4.434 ms** over 100 samples. A separately timed path starting with an already decoded CPU tensor and ending with a CPU output tensor measured mean **2.939 ms**, median **2.666 ms**, p95 **3.873 ms**. These distributions overlap; do not subtract the means to estimate transfer cost. Neither path includes PNG decode, Godot rendering, GPU texture readback, compositing, display, or GPU contention with the game.

We also attempted a direct matched renderer timing with the pinned Godot 4.7.2 app:

```sh
DLSS66_VARIANT=cheap DLSS66_TASK=shadow_resolution DLSS66_BENCHMARK=1 tools/Godot.app/Contents/MacOS/Godot --path . -- --dataset
```

The script reused one static scene and camera, alternated 512 and 4096 atlas sizes, warmed each condition for 40 frames, and sampled 100 frames with `RenderingServer.viewport_get_measured_render_time_gpu` and its CPU counterpart. GPU time was **0.0 ms on every sample**, including a second attempt with `RenderingServer.force_draw()` each frame. The CPU-only render-time counter varied between repeated blocks, so it cannot establish which atlas is faster. Godot's [4.7 API](https://docs.godotengine.org/en/4.7/classes/class_renderingserver.html) documents these counters, but this M2 Metal run did not yield usable GPU measurements. The measured counters exclude ML inference. A Metal GPU capture or another verified GPU timing method is needed. No matched 512-atlas vs 4096-atlas render-time result or end-to-end FPS exists yet.

During review, benchmark mode was found to open the default cheap manifest before branching, truncating that local file. The branch now runs before any manifest is opened. `sh collect_dataset.sh` regenerated and validated all 480 original pairs, and a subsequent benchmark left `manifest_cheap.jsonl` at 480 lines. No generated dataset files are committed.

### Validation-calibrated correction and new unseen scenes, 2026-11-14

The raw refinement model's mask still produced false corrections. [`evaluation/calibrate_mask.py`](../evaluation/calibrate_mask.py) tested logit biases 0, 0.5, 1, 1.5, 2, and 3 **on validation scenes 24–27 only** and chose bias **1.5** by lowest mean validation MAE (0.008138 versus 0.009529 at zero bias). No network weights changed. [`ShadowRefinementRenderer`](../models/multiscale.py) applies `sigmoid(mask_logits − 1.5)` at inference. On the already inspected test scenes 28–31, recalculation with the direct model forward pass yielded MAE **0.008868** versus cheap **0.011364**, PSNR **30.121** versus 27.344 dB, and improvement on **4/4 scenes and 42/48 frames**. Unchanged-pixel MAE was still worse than cheap (0.002238 versus 0.000645).

Because scenes 28–31 had already been inspected, we generated **four new scenes 32–35** into a separate local directory using the same fixed architecture and bias. The 48 fresh pairs passed the RGB/geometry validator: mean cheap/reference difference 2.630/255, 8.65% changed pixels, 0.791–9.332 m depth, normal-length error at most 1.2×10⁻⁷, zero sky-mask mismatch. **No fitting or calibration used these four scenes.**

| Fresh scenes 32–35 | Cheap 512 atlas | Calibrated model |
|---|---:|---:|
| Whole-image MAE ↓ | 0.010314 | **0.007360** |
| PSNR dB ↑ | 27.474 | **30.428** |
| Changed-pixel MAE ↓ | 0.112534 | **0.067825** |
| Unchanged-pixel MAE ↓ | **0.000591** | 0.001608 |
| Improved scenes / frames | — | **4/4; 47/48** |

The comparison at `artifacts/shadow_refine_fresh_comparison.png` was inspected for frame 0 of each scene. Shadows are sharper and mostly align with the high-atlas reference, but errors remain around contact edges. This establishes improvement within the current procedural scene family, not generalization to different game art, materials, or lighting. The increase in unchanged-pixel error means false appearance changes are reduced by calibration, not eliminated.

```sh
python3 -m evaluation.calibrate_mask --kind shadow_refine_mask --data dataset/shadow_resolution --checkpoint artifacts/shadow_refine_mask_32.pt --output artifacts/shadow_refine_calibration_32.json --device mps
DLSS66_SCENE_START=32 DLSS66_LAYOUTS=4 DLSS66_OUTPUT=dataset/shadow_resolution_fresh DLSS66_TASK=shadow_resolution sh collect_dataset.sh
python3 -m evaluation.final_holdout --device mps --data dataset/shadow_resolution_fresh --kind shadow_refine_mask --mask-bias 1.5 --checkpoint artifacts/shadow_refine_mask_32.pt --output artifacts/shadow_refine_holdout_fresh.json
python3 -m evaluation.visualize_refinement --data dataset/shadow_resolution_fresh --scenes 32 33 34 35 --mask-bias 1.5 --output artifacts/shadow_refine_fresh_comparison.png --device mps
```

### Offline temporal probe, 2026-11-14

The 12 camera poses form 30° steps around **static** scenes. [`evaluation/temporal.py`](../evaluation/temporal.py) reconstructs world positions from exported reverse-Z depth, reprojects them to the next camera, and rejects out-of-view and depth-inconsistent pixels (5 cm tolerance). An identity-camera reprojection unit test passed. The metric is the mean change in **prediction error relative to the reference** across corresponding visible surface pixels. About 51–56% of source pixels survived per scene. This is a controlled, offline proxy; it is not moving-object video or presentation timing.

On fresh scenes 32–35, the cheap renderer's error-change metric was **0.005065**; the calibrated model's was **0.007515**. The model is less temporally stable by this measure despite its better frame MAE. A depth-rejected history baseline then warped the previous output and blended 25% of it into the current output on visible surfaces. Blend strength was chosen from 0/0.25/0.5/0.75 using `validation MAE + validation temporal error`, without using fresh test scenes. On fresh scenes it lowered the model's temporal error from **0.007515 to 0.007243**, but increased image MAE from **0.007360 to 0.007780**. It still did not match the cheap renderer's temporal error. A real solution needs smoother trajectories, moving objects, engine motion vectors, disocclusion checks, and end-to-end latency measurements.

```sh
python3 -m evaluation.temporal --data dataset/shadow_resolution_fresh --device mps --mask-bias 1.5
python3 -m evaluation.temporal_blend --device mps --mask-bias 1.5
python3 -m unittest discover -s tests -q  # 11 tests passed
```
