# Stage 1 M2 buffer export — 2026-11-12

Machine: MacBook Air M2, 8 GB unified memory, macOS 26.5.1. Engine: pinned official Godot 4.7.2 stable (`ed1daf0bf`), native Metal 4.0 Forward+. Scene uses only procedural boxes.

## Reproduce

Place the official Godot 4.7.2 `Godot.app` in `tools/`. With Python, NumPy, and Pillow available, run from the repository root:

```sh
sh capture.sh
```

The script imports `export.glsl`, renders a static 256×144 scene, waits for five compositor frames, samples depth and normal/roughness in a compute pass, writes RGBA32F to an application-owned texture, requests asynchronous readback, saves RGB, then validates and visualizes the result. Outputs are in ignored `captures/`: `scene.png`, `geometry.rgba32f`, `depth_device.npy`, `depth_meters.npy`, `normal_view.npy`, `valid.npy`, `depth.png`, and `normal.png`.

## Observed results

- Direct readback of Godot's internal textures failed previously with error 31 because they lack `TEXTURE_USAGE_CAN_COPY_FROM_BIT`. The new owned texture has `STORAGE_BIT | CAN_COPY_FROM_BIT`, and `texture_is_format_supported_for_usage` passed on this M2. The depth and normal inputs both had sampling usage. `texture_get_data_async` returned `OK` and delivered exactly 589,824 bytes (= 256 × 144 × 4 × 4).
- Two consecutive `sh capture.sh` runs completed without Godot errors or RID leak warnings. Both reported 17,996 valid pixels of 36,864, foreground reverse-Z device depth 0.003679–0.019718, reconstructed linear depth 2.534–13.545 m, finite values, unit-length foreground normals, and zero background normals.
- RGB sky at the top center coincides with invalid depth; the floor at the bottom center has positive depth and a valid normal. The depth and normal visualizations show the same floor/box silhouettes and orientation as RGB.
- The two runs produced identical SHA-256 hashes: `scene.png` `9c1f6b7fc0369142cd597ea511c76581f0c3b863e426f111f4e62e5adadf0667`; `geometry.rgba32f` `34963b3a19cbcac789d28708e753ae88e00207366eb1432b92dd005cdd2f0fb3`.
- The reproducibility runs logged CPU compute-command submission of 3 and 2 microseconds, and 22.889 and 16.630 milliseconds from asynchronous readback request to callback. A later run logged 4 microseconds and 21.713 milliseconds. These are wall-clock observations, **not GPU kernel time or pure transfer time**. Godot `capture_timestamp()` markers were attempted but returned no export timestamp pair in this build (`EXPORT_GPU_TIMESTAMP_UNAVAILABLE`). A Metal GPU trace is required to separate GPU export execution and transfer precisely.

## Format and limits

`geometry.rgba32f` is row-major little-endian float32, 144×256×4: R = reverse-Z device depth; GBA = decoded view-space normal. The shader applies Godot's documented normal conversion `normalize(packed.xyz * 2 - 1)` and writes zero for background depth. `validate_capture.py` reconstructs positive view depth using the explicit camera near/far values (0.05/4000 m), writes `.npy` arrays, and rejects nonfinite/out-of-range/misaligned output. The visual depth PNG is percentile-scaled for inspection; use `depth_meters.npy` for numerical work.

Godot 4.7.2 did not create the normal/roughness texture from `needs_normal_roughness=true` alone in our earlier tests. `Environment.ssao_enabled=true` with intensity 0 creates it. This still invokes engine SSAO preparation and may add rendering cost; confirm its visual neutrality and measure that overhead before performance comparisons. The compute output is captured at `POST_OPAQUE`; the viewport RGB is read after the asynchronous callback while camera and scene are static. This verifies alignment for this static scene, not moving-frame temporal correspondence. First-frame capture produced empty depth, so the workflow waits five rendered frames. Cross-machine reproducibility and pixel-level physical correctness are not established.

## Next step

The [paired dataset](docs/DATASET.md) now captures RGB and cheap-pass depth/normals for deterministic camera poses across 16 procedural layouts. This does not measure GPU export and transfer separately; a Metal trace remains needed. Camera poses are static during each capture, and no moving-object temporal pipeline or Core ML integration has been added. Model experiments are recorded in [EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).
