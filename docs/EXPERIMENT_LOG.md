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

## Metal cost and live integration probe, executed 2026-11-15

The Godot viewport GPU timer remained `0.0` on this M2, so we used Xcode 16 `Metal System Trace` on the pinned Godot 4.7.2 app. [`dataset.gd`](../dataset.gd) now logs Unix phase timestamps and alternates conditions on one static scene and camera (scene 28 at 256×144), with 40 warmup and 100 measured draws per phase. [`evaluation/analyze_metal_trace.py`](../evaluation/analyze_metal_trace.py) assigns Godot GPU frames to phases using their timestamp midpoint, merges overlapping GPU-channel intervals per frame, and reports busy-time distributions. These are **GPU-work estimates**, not presented-frame latency. Full `.trace` files remain local in `/tmp` and are not committed. The table gives repeated phase medians, so variation is visible.

| GPU work per Godot frame | Cheap phase medians | Costly phase medians | Difference, approximate |
|---|---:|---:|---:|
| Shadow atlas 512 vs 4096 | 0.410, 0.451 ms | 0.655, 0.649 ms | 0.20–0.25 ms |
| Hard shadows vs PCSS at 0.5°; atlas 512 | 0.414, 0.428 ms | 0.491, 0.498 ms | 0.07–0.08 ms |
| SSIL off vs ultra on; atlas 512 | 0.438, 0.439 ms | 0.780, 0.789 ms | 0.34–0.35 ms |

These are three **separate traces**, not additive effects. The scene is small and the figures should not be generalized to games or higher resolutions. The official [Godot 4.7 RenderingServer API](https://docs.godotengine.org/en/4.7/classes/class_renderingserver.html) supplies the atlas and SSIL settings; [Godot's Light3D documentation](https://docs.godotengine.org/en/4.7/classes/class_light3d.html) identifies nonzero directional angular distance as PCSS with a performance cost. On this scene, none of these conventional effects saves enough GPU work to pay for the current 80,152-parameter PyTorch MPS model, whose standalone synchronized model-call median was about 2.3 ms. This is a serial-cost inference; overlapping or native GPU/ANE integration was not measured.

The Metal trace workflow was:

```sh
xcrun xctrace record --template 'Metal System Trace' --time-limit 25s --output /tmp/dlss66_metal_labeled.trace --target-stdout /tmp/dlss66_godot_bench.log --no-prompt --env DLSS66_VARIANT=cheap --env DLSS66_TASK=shadow_resolution --env DLSS66_BENCHMARK=1 --launch -- tools/Godot.app/Contents/MacOS/Godot --path . -- --dataset
xcrun xctrace export --input /tmp/dlss66_metal_labeled.trace --toc --output /tmp/dlss66_labeled_toc.xml
xcrun xctrace export --input /tmp/dlss66_metal_labeled.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' --output /tmp/dlss66_labeled_gpu_intervals.xml
python3 -m evaluation.analyze_metal_trace --toc /tmp/dlss66_labeled_toc.xml --intervals /tmp/dlss66_labeled_gpu_intervals.xml --log /tmp/dlss66_godot_bench.log
```

For PCSS or SSIL, add `--env DLSS66_BENCHMARK_EFFECT=pcss` or `ssil` to the `record` command and use distinct output paths. No Godot API errors were observed in these recordings.

We then ran a **24-frame external live bridge**: an isolated Godot `SubViewport` rendered scene 28 and exported RGB plus aligned depth/normals; Godot sent RGB8, RGBA32F geometry, and view-space light direction over local TCP; [`evaluation/serve_live.py`](../evaluation/serve_live.py) built the same 11 input channels as training, ran the calibrated model on MPS, and returned RGB8; Godot uploaded it as an `ImageTexture` and displayed it. The final displayed image was inspected at `artifacts/live_display.png`. The server compared each output with the matching scene-28 reference frame. Two orbits of the 12 camera poses gave mean cheap MAE **0.005956** and model MAE **0.005514**, with **18/24** displayed frames improved. These are repeated poses from one already inspected scene, so this is an integration and alignment check, not new generalization evidence.

After the first seven startup/transient frames, the remaining **17 frames** measured in `artifacts/live_godot.json` had median/p95 serial pipeline **55.17/69.69 ms** (about 18 FPS at the median). Component median times were geometry export and wait **27.30 ms**, RGB readback **9.54 ms**, local IPC plus inference **11.80 ms**, and output upload **0.059 ms**. The pipeline figure also includes an extra process-frame draw before returning. Server-side `artifacts/live_server.json` recorded median model transfer plus synchronized inference **6.75 ms** while the Godot app was active. A standalone exact-server benchmark with no Godot active measured median **7.62 ms** for condition assembly, transfer, inference, and RGB output. Making the input tensor contiguous reduced the standalone total from an earlier **9.69 ms** median; reusing a device input buffer gave no reliable further gain, so that extra state was removed. These timings vary with startup and system state. The bridge is a deliberately serial prototype, and its frame time is above both 16.7 ms (60 FPS) and 33.3 ms (30 FPS). No zero-copy or native integration is claimed.

Commands for the live check:

```sh
# Terminal 1, wait for LIVE_SERVER_READY:
python3 -m evaluation.serve_live --device mps --frames 24 --warmup 10
# Terminal 2:
DLSS66_VARIANT=cheap DLSS66_TASK=shadow_resolution DLSS66_LIVE_FRAMES=24 tools/Godot.app/Contents/MacOS/Godot --path . -- --live
python3 -m evaluation.benchmark_live_server --device mps --samples 100
```

This experiment answers the current cost question negatively for the **measured serial bridge**: the quality gain is measurable, but its processing overhead is far larger than the high-atlas GPU-work increment on this M2 scene. A matched high-atlas *presented-frame* timing was not measured, so this is not a quantified FPS speedup ratio. The offline temporal metric also remains worse than the cheap renderer. A next performance experiment needs either a genuinely expensive reference effect or a substantially cheaper native inference path; more training on the atlas task alone cannot create a performance advantage.

### Residual-history test on new scenes, 2026-11-15

[`evaluation/temporal_blend.py`](../evaluation/temporal_blend.py) now compares history of the entire RGB prediction with history of only the predicted correction (`prediction − cheap RGB`). It reprojects history with depth and rejects pixels that fail the existing visibility test. Candidate type and blend strength (0, 0.25, 0.5, 0.75) were selected on validation scenes 24–27 by the declared score `frame MAE + temporal error`. The selected setting was **correction history at 0.75**. The model and calibration bias were frozen.

We then generated 48 pairs in four new scenes **36–39**. The dataset validator reported 48 matched pairs, zero cheap/reference sky-mask disagreement, and normal-length error at most 1.2×10⁻⁷. These scenes were not used for training, calibration, or history selection. On them, the cheap renderer's image MAE was **0.009874** and the unblended model's was **0.007514** (47/48 frames and 4/4 scenes improved). The model's unchanged-pixel MAE remained worse: **0.001841** versus cheap **0.000669**.

| New scenes 36–39 | Cheap | Model | Correction history |
|---|---:|---:|---:|
| Image MAE ↓ | 0.009874 | **0.007514** | 0.007767 |
| Reprojected temporal error ↓ | **0.005108** | 0.007516 | 0.005727 |

Correction history improves temporal error by **23.8% relative to the unblended model**, with a **3.4% increase in model image MAE**. It does not yet match the cheap renderer's temporal stability. The measurement covers static scenes and 30° camera jumps only. It does not establish stability with moving objects, disocclusions in real gameplay, or real-time delivery. Dataset files and model checkpoints remain local.

```sh
DLSS66_SCENE_START=36 DLSS66_LAYOUTS=4 DLSS66_OUTPUT=dataset/temporal_test DLSS66_TASK=shadow_resolution sh collect_dataset.sh
python3 -m evaluation.final_holdout --device mps --data dataset/temporal_test --kind shadow_refine_mask --mask-bias 1.5 --checkpoint artifacts/shadow_refine_mask_32.pt --output artifacts/shadow_refine_holdout_temporal_test.json
python3 -m evaluation.temporal --data dataset/temporal_test --device mps --mask-bias 1.5
python3 -m evaluation.temporal_blend --device mps --mask-bias 1.5 --test-data dataset/temporal_test
```

### Smooth camera and moving sphere, 2026-11-15

To test temporal behavior beyond the earlier static 30° camera steps, [`dataset.gd`](../dataset.gd) can now capture 48 poses per orbit (7.5° steps) and move one procedural sphere along a fixed circular path. Its center and radius are saved per frame. Cheap and reference runs use identical scene seeds, camera positions, and sphere transforms; [`validate_dataset.py`](../validate_dataset.py) checks paired metadata. The new scenes **40–41** produced 96 valid pairs with no sky-mask disagreement. A separate repeat of scene 40, frame 0 produced **byte-identical cheap PNG, reference PNG, and RGBA32F geometry**. This proves repeatability for that sampled pose, not every frame or machine.

[`evaluation/temporal.py`](../evaluation/temporal.py) now translates reconstructed sphere surface points between known sphere centers before camera reprojection. It still rejects depth-inconsistent correspondences. This is an **offline known-transform proxy**, not Godot motion vectors or a complete moving-object optical-flow solution. The sphere accounts for only about 0.5–0.6% of frame pixels that pass the visibility test, so object-specific scores have limited coverage.

| Scenes 40–41, 96 frames | Cheap | Original model, weight 20/bias 1.5 | New model, weight 5/bias 1.0 |
|---|---:|---:|---:|
| Whole-frame image MAE ↓ | 0.012663 | **0.009307** | 0.009457 |
| Unchanged-pixel MAE ↓ | **0.000615** | 0.002227 | 0.001673 |
| Reprojected temporal error ↓ | **0.006195** | 0.007573 | 0.007177 |
| Improved frames over cheap | — | 90/96 | 89/96 |

The new model uses the **same 80,152-parameter architecture**, seed, 20 epochs, and 32 training scenes, changing only the L1 changed-pixel weight from 20 to 5. Its checkpoint was selected by validation MAE; mask bias **1.0** was also selected on validation scenes 24–27. On the previously unseen static scenes 36–39, it reduced MAE from cheap **0.009874** to **0.007264** (48/48 frames), compared with **0.007514** for the original model. Its unchanged-pixel MAE was **0.001341** versus **0.001841** for the original, but both still exceed cheap **0.000669**. On motion scenes it trades a small increase in image MAE for lower false-change and temporal errors. This does **not** establish absence of invented shadows.

For the new model, validation-selected correction history at strength 0.75 lowered motion-sequence temporal error further from **0.007177 to 0.006598**, while raising image MAE from **0.009457 to 0.009899**. The cheap renderer remained steadier at **0.006195**. The moving-surface temporal metric varied substantially between the two scenes and remains weaker evidence than the whole-frame metric. Visual inspection of `artifacts/shadow_refine_motion_comparison.png` shows residual errors around object shadows and contact edges. The live bridge and the model-only latency were not rerun with the new weights; its operator graph and tensor sizes are unchanged, but equivalent runtime remains an inference.

```sh
DLSS66_MOTION=1 DLSS66_SCENE_START=40 DLSS66_LAYOUTS=2 DLSS66_FRAMES=48 DLSS66_OUTPUT=dataset/motion_test DLSS66_TASK=shadow_resolution sh collect_dataset.sh
DLSS66_MOTION=1 DLSS66_SCENE_START=40 DLSS66_LAYOUTS=1 DLSS66_FRAMES=1 DLSS66_OUTPUT=dataset/motion_repeat DLSS66_TASK=shadow_resolution sh collect_dataset.sh
python3 -m training.train_pairs --kind shadow_refine_mask --changed-weight 5 --epochs 20 --data dataset/shadow_resolution --device mps --output artifacts/shadow_refine_mask_weight5_32.pt
python3 -m evaluation.calibrate_mask --kind shadow_refine_mask --data dataset/shadow_resolution --checkpoint artifacts/shadow_refine_mask_weight5_32.pt --output artifacts/shadow_refine_weight5_calibration.json --device mps
python3 -m evaluation.final_holdout --device mps --data dataset/motion_test --kind shadow_refine_mask --mask-bias 1.0 --checkpoint artifacts/shadow_refine_mask_weight5_32.pt --output artifacts/shadow_refine_weight5_motion_test.json
python3 -m evaluation.temporal --data dataset/motion_test --checkpoint artifacts/shadow_refine_mask_weight5_32.pt --device mps --mask-bias 1.0
python3 -m evaluation.temporal_blend --device mps --mask-bias 1.0 --checkpoint artifacts/shadow_refine_mask_weight5_32.pt --test-data dataset/motion_test
python3 -m unittest discover -s tests -q  # 13 tests passed
```

### Preserve unchanged pixels with a validation constraint, 2026-11-16

The previous comparison showed that `changed_weight=5, unchanged_weight=1` improved image quality but still introduced small dark corrections on unchanged pixels. [`training/train_pairs.py`](../training/train_pairs.py) now lets the two regions have explicit positive weights. `unchanged_weight=4` collapsed to near identity: validation MAE stayed about 0.01124 versus cheap 0.01123 after 20 epochs. A milder setting, `changed_weight=5, unchanged_weight=2`, learned a useful correction; its best validation checkpoint was epoch 20.

[`training/paired.py`](../training/paired.py) now reports a direct false-darkening measure: the share of reference-unchanged pixels where the model darkens cheap RGB by more than 5/255. On validation scenes 24–27, the balanced model's MAE-optimal mask bias was 1.0, with a 0.644% false-darkening rate. We added an optional validation constraint to [`evaluation/calibrate_mask.py`](../evaluation/calibrate_mask.py). With a declared limit of 0.5%, it selects bias **2.0** (validation MAE 0.007288; validation false-darkening 0.454%). This is a validation choice, not proof of zero hallucination.

The constraint transferred to new static scenes **36–39**: model MAE was **0.007012** versus cheap **0.009874**, all 48 frames improved, and false-darkening above 5/255 was **0.386%** of unchanged pixels. On moving scenes **40–41**, model MAE was **0.008614** versus cheap **0.012663**, all 96 frames improved; false-darkening was **0.636%**. These values are nonzero. The model's changed-pixel trade-off and small procedural object mean this does not establish general game behavior.

The offline reprojection temporal error at bias 2.0 was **0.006984** versus cheap **0.006195** on the moving sequence. Correction history strength 0.75, selected on validation scenes, reduced it to **0.006479** while increasing image MAE from **0.008614 to 0.009041**. This is closer to cheap but remains worse. The moving sphere contributes under 0.6% of visibility-valid pixels, and the score includes much larger static surfaces.

```sh
python3 -m training.train_pairs --kind shadow_refine_mask --changed-weight 5 --unchanged-weight 4 --epochs 20 --data dataset/shadow_resolution --device mps --output artifacts/shadow_refine_mask_balanced_32.pt
python3 -m training.train_pairs --kind shadow_refine_mask --changed-weight 5 --unchanged-weight 2 --epochs 20 --data dataset/shadow_resolution --device mps --output artifacts/shadow_refine_mask_balanced2_32.pt
python3 -m evaluation.calibrate_mask --kind shadow_refine_mask --data dataset/shadow_resolution --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_calibration_constrained.json --device mps --max-false-darken-rate 0.005
python3 -m evaluation.final_holdout --device mps --data dataset/temporal_test --kind shadow_refine_mask --mask-bias 2.0 --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_bias2_temporal_test.json
python3 -m evaluation.final_holdout --device mps --data dataset/motion_test --kind shadow_refine_mask --mask-bias 2.0 --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_bias2_motion_test.json
python3 -m evaluation.temporal --data dataset/motion_test --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --device mps --mask-bias 2.0
python3 -m evaluation.temporal_blend --device mps --mask-bias 2.0 --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --test-data dataset/motion_test
```

### Hard confidence gate, selected on validation, 2026-11-16

The `sigmoid(logit − bias)` mask still gives small nonzero corrections below its confidence threshold. [`ShadowRefinementRenderer`](../models/multiscale.py) now has an optional hard gate: pixels with logits below the bias receive exactly zero mask and therefore keep cheap RGB unchanged. This does not prevent false positives above the threshold. [`evaluation/calibrate_mask.py`](../evaluation/calibrate_mask.py) can calibrate the soft or hard gate with the same validation false-darkening limit.

With `changed_weight=5`, `unchanged_weight=2`, and hard gating, the validation-selected bias was **1.0**. Validation false darkening was 0.407% at this bias. On fresh static scenes 36–39, MAE was **0.006923** versus cheap **0.009874**, all 48 frames improved, and false darkening above 5/255 was **0.347%** of unchanged pixels. On moving scenes 40–41, MAE was **0.008472** versus cheap **0.012663**, all 96 frames improved, and false darkening was **0.577%**. Thus the measured threshold transfers closely to static scenes, but misses its limit on the moving sequence. False modifications above 2/255 were 0.803% and 0.948% respectively.

For this hard-gated model, offline temporal error was **0.007122** versus cheap **0.006195**. Validation-selected correction history at strength 0.75 reduced it to **0.006524**, with image MAE increasing from **0.008472 to 0.008874**. It remains less stable than cheap rendering. No history variant was measured with live Godot presentation. At this point, the earlier 55.17 ms serial result defined the external path; a later 48-frame geometry-conditioned run is recorded below.

```sh
python3 -m evaluation.calibrate_mask --kind shadow_refine_mask --data dataset/shadow_resolution --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_hard_calibration.json --device mps --max-false-darken-rate 0.005 --hard-gate
python3 -m evaluation.final_holdout --device mps --data dataset/temporal_test --kind shadow_refine_mask --mask-bias 1.0 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_hard_temporal_test.json
python3 -m evaluation.final_holdout --device mps --data dataset/motion_test --kind shadow_refine_mask --mask-bias 1.0 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_hard_motion_test.json
python3 -m evaluation.temporal --data dataset/motion_test --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --device mps --mask-bias 1.0 --hard-mask
python3 -m evaluation.temporal_blend --device mps --mask-bias 1.0 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --test-data dataset/motion_test
python3 -m unittest discover -s tests -q  # 14 tests passed
```

### Stricter hard-gate sweep, 2026-11-16

We expanded the validation mask-bias sweep from 0–3 to 0–10. With hard gating and a validation constraint of at most **0.1%** unchanged pixels darkened by more than 5/255, the lowest-MAE eligible bias was **4.0**. Validation MAE was 0.008802 versus cheap 0.011232, with a 0.090% false-darkening rate. Biases 5 and 6 reduced the rate further on validation (0.046% and 0.009%) but lost more image quality; bias 8 produced no corrections and no quality improvement.

The selected bias transferred to fresh static scenes 36–39: MAE **0.008391** versus cheap **0.009874**, 48/48 frames improved, and false darkening was **0.033%**. On moving scenes 40–41, MAE was **0.010432** versus cheap **0.012663**, 96/96 frames improved, and false darkening was **0.051%**. These figures are small but not zero. The hard gate ensures exact identity below its threshold; pixels above it can still be false positives.

At this threshold, no-history temporal error on the moving sequence was **0.006738** versus cheap **0.006195**. Correction-history strength 0.75, selected on validation by `MAE + temporal error`, lowered temporal error to **0.006340** and increased image MAE to **0.010657**. It also increased false darkening from **0.051% to 0.247%**, still below the declared 0.5% validation limit used for the blend experiment but above the stricter 0.1% gate target. History helps stability, yet does not beat cheap rendering and weakens shadow preservation. This is a clear trade-off rather than a completed temporal solution.

```sh
python3 -m evaluation.calibrate_mask --kind shadow_refine_mask --data dataset/shadow_resolution --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_hard_calibration_001.json --device mps --max-false-darken-rate 0.001 --hard-gate
python3 -m evaluation.final_holdout --device mps --data dataset/temporal_test --kind shadow_refine_mask --mask-bias 4.0 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_hard4_temporal_test.json
python3 -m evaluation.final_holdout --device mps --data dataset/motion_test --kind shadow_refine_mask --mask-bias 4.0 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_balanced2_hard4_motion_test.json
python3 -m evaluation.temporal --data dataset/motion_test --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --device mps --mask-bias 4.0 --hard-mask
python3 -m evaluation.temporal_blend --device mps --mask-bias 4.0 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --test-data dataset/motion_test
```

### RGB-only live bridge synchronization and M2 timing, 2026-11-16

The previous RGB-only live loop could begin returning the prior camera pose after sustained capture. The input RGB then stopped matching the paired scene frames, which invalidated its late-frame throughput measurements. [`dataset.gd`](../dataset.gd) now waits for two draw/process boundaries after each camera update before reading the `SubViewport` texture. [`evaluation/serve_live.py`](../evaluation/serve_live.py) saves selected received RGB frames after the timed run for alignment inspection.

The corrected run used Godot **4.7.2**, the Apple M2/Metal renderer, a 256×144 viewport, scene 28 and its 12-pose camera loop repeated four times. The model was `shadow_refine_rgb_balanced2_32.pt`, with mask bias 4 and a hard gate, running through PyTorch MPS. All 13 saved sample inputs (including frames 23, 24, 35–40, and 47) exactly matched their expected cheap-render PNGs byte-for-byte. Across all 48 frames, cheap-input MAE against the paired reference averaged **0.005956**; model MAE averaged **0.005857**, with the model lower on **32/48 frames**. This is a small gain on one held-out procedural layout; it is not broad visual validation.

Steady timing excludes the first 12 startup frames. Latencies are serial and include the two Godot render boundaries needed to keep camera and RGB aligned:

| Stage (36 steady frames) | p50 | p95 |
|---|---:|---:|
| Camera update and render boundaries | 16.88 ms | 20.96 ms |
| RGB readback | 1.25 ms | 1.89 ms |
| IPC, MPS inference and response | 8.52 ms | 10.99 ms |
| Output texture upload | 0.06 ms | 0.09 ms |
| Full displayed-frame pipeline | **27.70 ms** | **35.50 ms** |

The server-side model call plus MPS synchronization had a 6.96 ms median and 10.05 ms p95. These measurements include CPU-to-MPS input transfer before the model and MPS-to-CPU output transfer afterward. The bridge uses a CPU socket path, so it does not measure zero-copy inference. The steady median corresponds to about 36 serial frames per second, while the p95 corresponds to about 28; it does not meet a stable 60 FPS budget.

The primary model also uses depth, normals and light direction. The first attempt at a 48-frame 11-channel run failed before inference because Godot timed out while connecting to the local server. A later one-frame probe and full 48-frame rerun completed successfully; the result is in the next section. The timeout was intermittent and its cause remains unknown.

```sh
python3 -m evaluation.serve_live --checkpoint artifacts/shadow_refine_rgb_balanced2_32.pt --kind shadow_refine_rgb --device mps --frames 48 --warmup 10 --mask-bias 4 --hard-mask
DLSS66_VARIANT=cheap DLSS66_TASK=shadow_resolution DLSS66_LIVE_RGB_ONLY=1 DLSS66_LIVE_FRAMES=48 DLSS66_OUTPUT=dataset/shadow_resolution tools/Godot.app/Contents/MacOS/Godot --path . -- --live
PYTHONPATH=. python3 -m unittest discover -s tests -q  # 16 tests passed; CPU training smoke test also passed
```

### Full geometry-conditioned live path, 2026-11-16

The 11-channel model was then run through the same 48-frame scene-28 camera loop with its RGB, RGBA32F depth/normal export, view-space light direction and hard-gated bias-4 checkpoint. After two earlier local-connect timeouts, a one-frame probe connected and completed; the subsequent 48-frame run also completed. The earlier timeout was intermittent and its root cause is not known. The final run contained 48 server measurements and 48 Godot measurements. All 13 sampled received RGB images (including frames 23, 24, 35–40 and 47) exactly matched the expected cheap-render PNGs, so no late camera-pose drift was present.

On these repeated integration poses, mean RGB MAE was **0.005956** for cheap rendering and **0.005544** for the model; the model was lower on **32/48 frames**. This checks that the complete geometry-conditioned interface produces plausible outputs. It does not add new generalization evidence beyond the held-out-scene offline tests because it repeats scene 28.

The latest run added stage timers to [`dataset.gd`](../dataset.gd). The following are p50/p95 over frames 2–47 to exclude cold startup:

| Host-observed stage | p50 | p95 |
|---|---:|---:|
| Camera update and two render/process boundaries | 66.60 ms | 67.06 ms |
| Wait for compositor geometry export callback | 33.41 ms | 33.73 ms |
| RGB texture readback | 18.15 ms | 19.82 ms |
| Godot request/response, MPS inference and return | 15.29 ms | 15.82 ms |
| Output upload | 0.06 ms | 0.11 ms |
| Full serial pipeline, including output presentation | **166.58 ms** | **167.22 ms** |

Server-side input preparation had 1.67 ms median / 2.91 ms p95. Model transfer to MPS, forward pass and synchronization took 6.81 ms median / 8.08 ms p95. These values show that the CNN call is not the main bottleneck. Pose synchronization, waiting for the geometry callback, RGB readback and output presentation dominate. They are wall-clock stage durations; they do not isolate the GPU time of the export shader. The external serial path sustains only about 6 FPS, so it is not real-time. A direct GPU-native path is needed before making a positive real-time claim.

The successful one-frame probe and 48-frame rerun both used:

```sh
python3 -m evaluation.serve_live --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --kind shadow_refine_mask --device mps --port 8766 --frames 48 --warmup 10 --mask-bias 4 --hard-mask --accept-timeout 300
DLSS66_VARIANT=cheap DLSS66_TASK=shadow_resolution DLSS66_LIVE_FRAMES=48 DLSS66_OUTPUT=dataset/shadow_resolution tools/Godot.app/Contents/MacOS/Godot --path . -- --live
```

### Motion-aware temporal-loss fine-tune, 2026-11-17

The single-frame model had better held-out frame MAE but worse temporal error than the cheap renderer. [`training/train_temporal.py`](../training/train_temporal.py) fine-tunes the same 80,152-parameter masked model with consecutive frame pairs. It adds a depth-tested reprojection loss on the model's error relative to the reference. Training used the existing static training split plus four moving training layouts (452 paired frames total); selection used static validation plus four separate moving validation layouts (48 static pairs, 192 moving pairs, and 188 transitions). The 96 frames in moving scenes 40–41 remained held out.

The best checkpoint was epoch 5. On moving validation, frame MAE fell from **0.007853 to 0.007343**, and the reprojection temporal error fell slightly from **0.007023 to 0.006879**. On the separate moving test scenes, at the same hard mask bias 4, fine-tuning lowered image MAE from **0.010432 to 0.010165** (all 96 frames improved over cheap input) and raised false darkening from **0.0506% to 0.0635%**, still under the 0.1% target. On the fresh static holdout scenes 36–39, MAE was **0.008152** versus **0.009874** for cheap input; all 48 frames improved, with false darkening **0.0343%**. Compared with the original checkpoint on those same static scenes, fine-tuning also reduced MAE from **0.008392 to 0.008152**.

Temporal generalization did not improve. On moving test scenes 40–41, the original model's reprojection temporal error was **0.006738** and the fine-tuned model's was **0.006795**; the cheap renderer was **0.006195**. A history correction blend selected only on moving validation scenes (bias 4, residual blend 0.25 under the unchanged-pixel darkening cap) lowered test temporal error to **0.006598** and kept false darkening at **0.0836%**. It still trails cheap rendering. A separate disagreement-aware blend sweep made a smaller gain (0.006795 to 0.006732 on test) and was not added to the code. These are offline, depth-tested known-transform reprojection metrics, not live temporal integration or engine motion-vector results.

This result supports the spatial model's held-out image improvement, but rejects the claim that adding a single-frame reprojection loss is enough to solve temporal consistency. The next temporal model should receive the reprojected previous correction as an explicit input and be trained/evaluated recurrently, including its own prior predictions to measure rollout drift. The current renderer bridge is still too slow for real-time use (166.58 ms median for the 11-channel path); model changes alone cannot remove its geometry and RGB readback waits.

Commands for the motion-aware fine-tune and the matched held-out comparison:

```sh
python3 -m training.train_temporal --data dataset/shadow_resolution --motion-data dataset/motion_train --validation-data dataset/motion_validation --init-checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_temporal_motion4_32.pt --device mps --epochs 5 --batch-size 2 --temporal-weight 0.5
python3 -m evaluation.final_holdout --data dataset/temporal_test --kind shadow_refine_mask --mask-bias 4 --hard-mask --checkpoint artifacts/shadow_refine_temporal_motion4_32.pt --device mps --output artifacts/temporal_motion4_static_holdout_bias4.json
python3 -m evaluation.final_holdout --data dataset/motion_test --kind shadow_refine_mask --mask-bias 4 --hard-mask --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --device mps --output artifacts/base_motion_holdout_bias4.json
python3 -m evaluation.final_holdout --data dataset/motion_test --kind shadow_refine_mask --mask-bias 4 --hard-mask --checkpoint artifacts/shadow_refine_temporal_motion4_32.pt --device mps --output artifacts/temporal_motion4_motion_holdout_bias4.json
python3 -m evaluation.temporal --data dataset/motion_test --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --device mps --mask-bias 4 --hard-mask
python3 -m evaluation.temporal --data dataset/motion_test --checkpoint artifacts/shadow_refine_temporal_motion4_32.pt --device mps --mask-bias 4 --hard-mask
python3 -m evaluation.temporal_blend --validation-data dataset/motion_validation --test-data dataset/motion_test --checkpoint artifacts/shadow_refine_temporal_motion4_32.pt --device mps --mask-bias 4 --hard-mask --max-false-darken-rate 0.001 --output artifacts/temporal_motion4_bias4_safe.json
PYTHONPATH=. python3 -m unittest discover -s tests -q
```

The paired holdout and temporal metric commands were repeated in the sandboxed execution environment on CPU; they matched earlier MPS results closely. At that time, the sandboxed Python runtime reported MPS unavailable. A later out-of-sandbox probe showed that the installed PyTorch 2.13.0 build has MPS support and that the sandbox was hiding the device. The MPS recurrent training, evaluation, and latency results are recorded below. The full unit suite passed: **21 tests**, plus the two-step CPU synthetic training smoke test. Local motion datasets and checkpoints remain ignored by Git.

### Explicit recurrent conditioning, 2026-11-17

The previous experiment's loss changed a single-frame network but gave it no prior-frame input. [`TemporalShadowRefinementRenderer`](../models/temporal.py) now takes 15 channels: current RGB, depth, normals, geometry-valid mask and light direction (11); previous predicted RGB correction after depth-tested reprojection (3); and a correspondence-valid mask (1). Its first convolution copies the spatial checkpoint weights and zero-initializes the four new input slices. A unit test confirms that arbitrary history initially leaves the spatial model output unchanged. The model has **80,584 parameters**, 432 more than the spatial model.

[`training/train_recurrent.py`](../training/train_recurrent.py) trains on three-frame rollouts. The first frame uses empty history. The next two frames use the model's own prior predicted correction, reprojected into the current view. The correction is detached between steps, so this is truncated recurrent training rather than full backpropagation through time. Supervised weighted L1, mask BCE and depth-tested error-consistency loss are applied across the three predictions. [`evaluation/evaluate_recurrent.py`](../evaluation/evaluate_recurrent.py) evaluates full sequences with the model's own outputs fed back. This exposes rollout drift that a shuffled single-frame metric cannot measure.

Initial recurrent training used CPU because the sandbox reported MPS unavailable. The five-epoch run with temporal loss weight 0.5 selected epoch 2. A three-epoch run with weight 2.0 also selected epoch 2 and gave slightly better validation values. Bias 4 failed the 0.1% moving validation false-darkening cap; bias 4.25 passed at 0.0937%, and had lower frame MAE than the next tested safe bias, 4.5. This threshold was fixed before running the held-out test sequences.

At validation bias 4.25, the stronger temporal-loss model produced:

| Split | Cheap frame MAE | Recurrent frame MAE | Cheap temporal error | Recurrent temporal error | False darkening |
|---|---:|---:|---:|---:|---:|
| Static validation, scenes 24–27 | 0.011232 | **0.008169** | 0.005209 | 0.006722 | 0.0829% |
| Moving validation, scenes 24–27 | 0.011931 | **0.008750** | 0.006042 | 0.006725 | 0.0937% |
| Static holdout, scenes 36–39 | 0.009874 | **0.007956** | 0.005108 | 0.006131 | 0.0486% |
| Moving holdout, scenes 40–41 | 0.012663 | **0.009766** | 0.006195 | 0.006856 | 0.0922% |

Frame quality improved on both held-out sets, with false darkening below the 0.1% cap. Temporal error did not: it is about **20.0% higher** on the static holdout and **10.7% higher** on the moving holdout than the cheap renderer. A same-checkpoint validation ablation found that recurrent history slightly reduced frame MAE, but did not reduce temporal error versus running the same weights without history. Raising temporal loss weight from 0.5 to 2.0 changed validation temporal error only from about 0.00681 to 0.00675. This recurrent design therefore does not meet the temporal goal. The earlier CPU training is superseded by the MPS run below; live Godot integration remains unmeasured for the recurrent model.

Commands for the full training and held-out rollout evaluation:

```sh
OMP_NUM_THREADS=4 python3 -m training.train_recurrent --data dataset/shadow_resolution --motion-data dataset/motion_train --validation-data dataset/motion_validation --init-checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_recurrent_m2_temporal2.pt --device cpu --epochs 3 --batch-size 1 --temporal-weight 2.0
OMP_NUM_THREADS=4 python3 -m evaluation.evaluate_recurrent --checkpoint artifacts/shadow_refine_recurrent_m2_temporal2.pt --data dataset/temporal_test --split test --output artifacts/recurrent_static_test.json --device cpu --mask-bias 4.25
OMP_NUM_THREADS=4 python3 -m evaluation.evaluate_recurrent --checkpoint artifacts/shadow_refine_recurrent_m2_temporal2.pt --data dataset/motion_test --split test --output artifacts/recurrent_motion_test.json --device cpu --mask-bias 4.25
PYTHONPATH=. python3 -m unittest discover -s tests -q
```

The 24-test suite passed after adding the recurrent model, including parameter-transfer, arbitrary-history initialization, tensor-shape, visibility-mask, temporal-loss and candidate-selection checks. The full Godot bridge remains at 166.58 ms median on the 11-channel MPS path; this recurrent external route has not been profiled, and CPU-only model experiments do not establish real-time feasibility.

### Stronger temporal objective and safe-history check, 2026-11-17

Two more short CPU training runs tested whether a stronger consistency loss could close the temporal gap. The 3-epoch run at `temporal_weight=10` selected epoch 1 by its validation objective. It retained the hard-gate bias 4, which was fixed before test evaluation and measured **0.0696%** false darkening on moving validation. Compared with `temporal_weight=2`, its validation temporal error fell from **0.006754 to 0.006470**; the cheap validation value was **0.006042**. Frame MAE was 0.009004 on moving validation versus 0.011931 for cheap input.

On the untouched scenes 36–39 and 40–41, this checkpoint reduced frame MAE from **0.009874 to 0.008103** (static) and **0.012663 to 0.010088** (moving). False darkening was **0.0321%** and **0.0755%**, respectively. Temporal error was still worse than the cheap renderer: **0.005739 vs 0.005108** on static scenes and **0.006642 vs 0.006195** on moving scenes. The stronger loss reduced temporal error relative to the weight-2 checkpoint, but did not solve the gap.

On the weight-2 recurrent checkpoint, we also tested a post-model blend of current correction with the reprojected previous correction (`alpha` = 0, 0.25, 0.5, 0.75) on validation. Candidate selection used mean frame MAE plus mean temporal error, subject to a maximum 0.1% false-darkening rate on both static and moving validation. `alpha=0` was the only eligible candidate: `alpha=0.25` exceeded the cap on moving validation (0.125%), and larger blends exceeded it further. The filter therefore could not be applied safely under this preservation constraint. The weight-10 test set was not used to select checkpoint or threshold, and no test sequence was used to select blend strength.

This experiment reinforces the main trade-off: stronger temporal supervision and history can lower flicker, but shadow correction can spill onto pixels that the reference says should remain unchanged. The next model change should predict a confidence/uncertainty for reprojected history and train it against visibility and per-pixel correction error, rather than blending history uniformly. The 25-test suite passed after fixing a blend-test expectation. These initial runs used M2 CPU because the sandbox hid MPS; an out-of-sandbox MPS training run and benchmark follow.

Reproduction commands:

```sh
OMP_NUM_THREADS=4 python3 -m training.train_recurrent --data dataset/shadow_resolution --motion-data dataset/motion_train --validation-data dataset/motion_validation --init-checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_recurrent_m2_temporal10.pt --device cpu --epochs 3 --batch-size 1 --temporal-weight 10.0
OMP_NUM_THREADS=4 python3 -m evaluation.evaluate_recurrent --checkpoint artifacts/shadow_refine_recurrent_m2_temporal10.pt --data dataset/temporal_test --split test --output artifacts/recurrent_static_test.json --device cpu --mask-bias 4
OMP_NUM_THREADS=4 python3 -m evaluation.evaluate_recurrent --checkpoint artifacts/shadow_refine_recurrent_m2_temporal10.pt --data dataset/motion_test --split test --output artifacts/recurrent_motion_test.json --device cpu --mask-bias 4
PYTHONPATH=. python3 -m unittest discover -s tests -q
```

### MPS training, held-out inference, and latency, 2026-11-17

The MPS issue was environmental. Both installed PyTorch builds reported `is_built() == True` but `is_available() == False` inside the sandbox. An approved probe outside the sandbox returned `is_available() == True` and successfully created and summed an MPS tensor. The same PyTorch 2.13.0 Python 3.13 installation was used for the following run.

The recurrent model was trained on MPS for 3 epochs with temporal loss weight 10; validation selected epoch 1. At hard-gate bias 4, validation motion MAE was **0.008992** versus cheap **0.011931**, and temporal error was **0.006473** versus cheap **0.006042**. This matched the CPU run closely. The checkpoint was then evaluated on untouched held-out scenes:

| Split | Cheap frame MAE | MPS recurrent MAE | Cheap temporal error | MPS recurrent temporal error | False darkening |
|---|---:|---:|---:|---:|---:|
| Static scenes 36–39 | 0.009874 | **0.008095** | 0.005108 | 0.005747 | 0.0329% |
| Moving scenes 40–41 | 0.012663 | **0.010072** | 0.006195 | 0.006646 | 0.0766% |

Thus MPS inference confirms the CPU image-quality and temporal trade-off. Frame MAE improves by about 18% static and 20% moving; temporal error remains about 12% and 7% higher than cheap rendering. No test sequence was used for checkpoint or threshold selection.

[`evaluation/benchmark_recurrent.py`](../evaluation/benchmark_recurrent.py) measured 100 synchronized samples at 256×144, after 10 warmups:

| Recurrent inference path | p50 | p95 |
|---|---:|---:|
| Model only, tensors already on MPS | 2.43 ms | 2.69 ms |
| CPU-to-MPS tensor handoff | 1.01 ms | 1.09 ms |
| History warp plus model | 2.60 ms | 3.36 ms |
| MPS-to-CPU output readback | 0.19 ms | 0.22 ms |
| Handoff, warp, model, and readback | **3.80 ms** | **4.63 ms** |

The benchmark uses synthetic CPU tensors and an identity reprojection grid. It excludes Godot rendering, real geometry export, geometry-readback wait, IPC and presentation. It is not an end-to-end frame rate. The earlier 11-channel Godot bridge measured **166.58 ms median**; the recurrent path has not yet been integrated into that bridge. The current evidence shows the model compute is small enough for a real-time budget, while end-to-end feasibility remains open because renderer synchronization and readback dominate.

Reproduction commands:

```sh
python3 -m training.train_recurrent --data dataset/shadow_resolution --motion-data dataset/motion_train --validation-data dataset/motion_validation --init-checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output artifacts/shadow_refine_recurrent_m2_mps_temporal10.pt --device mps --epochs 3 --batch-size 1 --temporal-weight 10.0
python3 -c "from pathlib import Path; from evaluation.evaluate_recurrent import run; run(Path('artifacts/shadow_refine_recurrent_m2_mps_temporal10.pt'), Path('dataset/temporal_test'), 'test', Path('artifacts/recurrent_mps_static_test.json'), 'mps', 4.0, True); run(Path('artifacts/shadow_refine_recurrent_m2_mps_temporal10.pt'), Path('dataset/motion_test'), 'test', Path('artifacts/recurrent_mps_motion_test.json'), 'mps', 4.0, True)"
python3 -m evaluation.benchmark_recurrent --checkpoint artifacts/shadow_refine_recurrent_m2_mps_temporal10.pt --device mps --width 256 --height 144 --warmup 10 --samples 100 --output artifacts/recurrent_mps_benchmark.json
PYTHONPATH=. python3 -m unittest discover -s tests -q
```
