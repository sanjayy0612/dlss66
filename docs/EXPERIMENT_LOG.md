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

`sh collect_dataset.sh` ran Godot 4.7.2 on the M2 Metal Forward+ renderer. It captured 96 pairs of 256×144 RGB PNGs from eight procedural scene layouts and 12 camera poses per layout. The two passes differed by directional shadows off versus on. `validate_dataset.py` checked matching seed, camera pose, dimensions, file presence, unique keys, and nonzero image differences. The scene split is 72 train, 12 validation, and 12 test pairs. Mean absolute cheap/reference difference was 10.199/255 per channel; the smallest pair was 2.041/255 and the largest 18.297/255. On average, 12.31% of pixels differed by more than five levels in at least one channel. A four-pair pilot produced identical SHA-256 PNG hashes on two runs. One sampled PNG pair and the manifest had matching hashes across two full runs. Full-corpus hash reproducibility and geometry-mask alignment were not independently measured. Details: [DATASET.md](DATASET.md).

## Planned, not executed

- Repeat full-corpus hashes and verify object-mask alignment if this dataset is used for a quantitative training claim.
- Add aligned depth, normals, and a light-direction input; increase layout diversity before interpreting failed RGB-only shadow prediction as a general limit of neural rendering.
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
