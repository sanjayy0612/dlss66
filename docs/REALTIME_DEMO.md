# Real-time Godot demo: implementation path

## Goal

Show the same moving Godot scene with the cheap shadow, a learned refinement, and the high-quality reference. The learned result must run in the renderer's GPU path. The comparison must report image quality, motion stability, and full-frame cost.

The live spatial model now runs in Godot's Forward+ compositor on an Apple M2. It receives renderer color, depth, normals, and view-space light direction. The current path has no temporal history, and end-to-end GPU frame time is not measurable through the tested Metal timing counters. The older Python bridge took 166.58 ms median for geometry-conditioned inference and transfer; that number does not describe the new GPU path.

## First artifact: moving-scene video

[`neural-shadow-demo.mp4`](videos/neural-shadow-demo.mp4) shows 48 held-out frames from moving scene 48. It was rendered offline from recorded RGB, geometry, and known scene motion. The neural result is not live Godot inference. Regenerate it with:

```sh
python3 -m evaluation.make_demo_video --scene 48 --count 48 --fps 12 --output docs/videos/neural-shadow-demo.mp4
```

This clip is useful for judging visible changes. It does not establish real-time performance or solve temporal error.

## Live Godot GPU result

[`neural-shadow-godot-gpu.mp4`](videos/neural-shadow-godot-gpu.mp4) is a 25-frame, 2.1-second Movie Maker capture from the live compositor. The recorded display is 1152×648 at 12 FPS. The model input and network output are 256×144; Godot's display scaling expands the viewport. The inference path stays on the GPU and does not read renderer textures back to the CPU each frame.

The compositor packs the current color, depth, normal, and camera-space light direction into 11 planar channels. It runs the exported spatial U-Net in compute passes, then writes the predicted RGB into the active Godot color image. Press **N** in interactive mode to toggle the effect. Temporal history and motion-vector reprojection are not implemented.

The same-pose [comparison image](images/neural-shadow-gpu-live-comparison.png) and capture metrics come from 12 frames of a fixed camera and a moving sphere. The model runs independently on each frame. Relative to the cheap 512-shadow render, it lowered mean frame MAE against the 4096-shadow reference by **4.3%** overall and **8.2%** in changed shadow regions. On unchanged pixels, mean error rose from **0.00079** to **0.00113**. Static-region temporal error rose from **0.00023** to **0.00040**. This one small scene shows a shadow improvement with a stability cost.

Reproduce the GPU video (the AVI is a temporary Movie Maker output, then encoded as H.264):

```sh
DLSS66_GPU_DEMO_FRAMES=16 tools/Godot.app/Contents/MacOS/Godot --path . --write-movie /tmp/neural-shadow-godot.avi --fixed-fps 12 -- --neural-gpu --record-neural-gpu
ffmpeg -y -i /tmp/neural-shadow-godot.avi -an -c:v libx264 -pix_fmt yuv420p docs/videos/neural-shadow-godot-gpu.mp4
```

Reproduce the matched quality and stability capture:

```sh
DLSS66_GPU_PAIR_FRAMES=12 tools/Godot.app/Contents/MacOS/Godot --path . -- --neural-gpu --capture-gpu-pairs
python3 -m evaluation.evaluate_gpu_capture --capture-dir artifacts/gpu_shadow_pairs --output artifacts/gpu_shadow_pairs/metrics.json
```

These capture tools read images to the CPU for offline scoring. The normal renderer path does not. The Godot GPU timestamps and viewport GPU-time counters returned zero on this Metal run, so the logs report compositor callback CPU time but cannot provide a trustworthy GPU p50/p95. A recorded-input end-to-end check, including explicit output readback, took 36.3 ms once; this single sample is not a live frame-time measurement.

## GPU implementation steps

1. **Export model weights — bundled.** The demo includes the 80,152-parameter runtime model in `godot/model/` as little-endian float32 tensors plus a JSON manifest. Godot loads these bytes into a storage buffer without importing PyTorch. Use the exporter below to regenerate them from the training checkpoint and create a PyTorch validation input.

   ```sh
   python3 -m evaluation.export_godot_weights --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output-dir artifacts/godot_model --mask-bias 4
   ```

   The desktop GPU validation command is `tools/Godot.app/Contents/MacOS/Godot --path . -- --gpu-weights`. Headless mode has no renderer-owned `RenderingDevice`. This probe uses one recorded input and checks the full spatial network against PyTorch.
2. **Read renderer inputs on the GPU — implemented.** The compositor samples color, depth, and normals and packs the channel transforms expected by training. It does not read those textures to the CPU each frame.
3. **Integrate the spatial network — implemented.** The convolution, pooling, resize, and skip operations run in GPU compute passes. The recorded-input output matches PyTorch, and the live output is written to the active renderer color image.
4. **Display and compare — first pass complete.** The `N` key toggles refinement. A live GPU video and a matched cheap/model/reference capture are available. The model improves shadow pixels slightly on the tested scene but worsens unchanged-pixel and temporal error.
5. **Measure the whole frame — incomplete.** Metal returned zero for Godot GPU timestamps and viewport GPU time. Callback CPU timing is not GPU latency. Use a backend with working GPU timestamps or an external GPU profiler before claiming a frame budget.
6. **Add temporal history — incomplete.** Reprojection must use engine motion data and reject disoccluded pixels. The offline known-sphere warp is not proof that engine motion vectors work.

## Verified GPU milestone

On the Apple M2 with Godot 4.7.2 Metal Forward+, the checkpoint weights and a full spatial-model forward pass run in Godot compute shaders. The input is one recorded held-out frame from scene 48. The GPU output matched PyTorch with **2.44×10⁻⁶ maximum absolute error** and **3.05×10⁻⁹ mean absolute error** over 110,592 RGB values. The measured compute, command submission, and output readback path took **34.1 ms** for this one run.

Reproduce the recorded-frame check:

```sh
python3 -m evaluation.export_godot_weights --checkpoint artifacts/shadow_refine_mask_balanced2_32.pt --output-dir artifacts/godot_model --mask-bias 4 --data dataset/expanded_motion_test --probe-scene 48 --probe-frame 0
tools/Godot.app/Contents/MacOS/Godot --path . -- --gpu-weights
```

This proves the GPU shader math and weight layout for the spatial network. The live compositor integration now packs current renderer textures and displays the result without per-frame CPU texture readback. It still does not prove a real-time effect: GPU frame timing is unavailable on this Metal path, and the model has no temporal history.

## Completion gates

- The model runs in the Godot GPU path without per-frame CPU texture readback.
- Godot output matches the Python reference at the same resolution and inputs.
- Camera and object motion do not produce unacceptable flicker or false darkening on held-out scenes.
- Full-frame p50/p95 cost and quality are measured against cheap and conventional rendering.
- The demo and video state whether inference is live or offline.

If the spatial GPU implementation is too slow, profile individual passes and reduce work based on measurements. Do not claim a real-time result based on model-only timing.
