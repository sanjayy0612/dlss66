# Proposed architecture

Historical AO-centered proposal (2026-11-11). The current ML-first direction and implemented RGB baseline are in [MODEL_DESIGN.md](MODEL_DESIGN.md). AO remains an optional classical ML baseline.

Research snapshot: 2026-11-11. This is a proposed implementation, not a measured system.

## One recommended path

Use a pinned **Godot 4 Forward+** demo with openly licensed assets. Add one small, explicit screen-space ambient-occlusion (AO) pass that can run at cheap and reference sample counts. Export matched buffers offline. Train a tiny geometry-conditioned residual CNN to predict the high-sample AO value from cheap AO, linear depth, and view-space normals. Composite the predicted scalar AO only into the ambient-light term. This bounded output protects object color and identity more directly than unrestricted RGB generation. Later add velocity-guided history, then consider live Core ML integration.

`fixed scene + camera + seed → geometry/depth/normals → cheap AO + cheap shaded RGB → offline buffer capture → CNN AO estimate → ambient composite → output`

For the paired target, replay the same state with a 32–64-sample AO pass. Disable temporal accumulation, auto exposure, stochastic lighting and other history effects during first pairing. Save all transforms, renderer settings, shader revision and seed with each sample. Capture at 256×144 first; 320×180 only if memory and quality justify it. These sample counts are experiment knobs, not validated performance targets.

## Buffer contract

| Data | Inference? | Proposed source | Format/validation |
|---|---|---|---|
| Cheap AO | Yes | Our custom AO render/compute pass output texture | R16F or R8; verify [0,1] and sign |
| Linear depth | Yes | `get_texture("render_buffers", "depth")`, then explicit linearization using projection parameters | R16F; verify near/far mapping and background mask |
| View normals | Yes | `get_texture("forward_clustered", "normal_roughness")` with `needs_normal_roughness=true` | Decode per pinned engine version, normalize, inspect visualization |
| Cheap RGB | Evaluation and later optional input | resolved color texture / viewport capture | consistent linear versus display color space |
| Reference AO/RGB | Training/evaluation only | matched high-sample pass and its composite | same scene/camera/seed; no history |
| Velocity | Temporal stage only | `get_velocity_texture()` with `needs_motion_vectors=true` | determine units/direction and invalid pixels by moving-object probe |
| Validity/disocclusion | Temporal stage | depth/normal/velocity consistency check written by us | disallow history at uncovered pixels |

The above texture names and flags come from the [stable API reference](https://docs.godotengine.org/en/stable/classes/class_compositoreffect.html). Godot's rendering API is experimental; Stage 1 must validate the exact buffers in a pinned release. Capture through `Viewport.get_texture().get_image()` is a simple offline route, with readback cost measured separately. A custom compute pass may need explicit texture-to-image export; no zero-copy Python bridge is assumed.

**Stage 1 update (2026-11-12):** Godot 4.7.2 on M2 was tested. Internal depth/normal textures are sampleable but lack `CAN_COPY_FROM_BIT`, so direct readback fails. `export.glsl` samples them into an owned RGBA32F texture with `STORAGE_BIT | CAN_COPY_FROM_BIT`; `texture_get_data_async` successfully returns 589,824 bytes at 256×144. `validate_capture.py` decodes raw reverse-Z depth and packed view normals into numerical arrays and visualizations. The normal texture appeared only when SSAO was enabled at zero intensity, despite `needs_normal_roughness=true`; this workaround has unmeasured cost. See [TEST_RESULTS.md](../TEST_RESULTS.md). GPU export time has not yet been isolated from readback/synchronization delay.

## Model and data

Initial model: 5-channel input (AO 1, depth 1, normal XYZ 3), 3×3 convolution 5→16, four 3×3 16→16 residual/ReLU blocks, 3×3 16→1, bounded AO output. Rough parameter target: 10–20 thousand, subject to exact block implementation. Compare with a non-neural depth/normal-aware bilateral AO filter and with raising AO sample count within the same time budget. Train with L1 or Charbonnier error on AO plus a modest edge-weighted term; validate RGB after ambient compositing. Avoid adversarial/perceptual losses until basic reconstruction and geometry preservation are established. A tiny U-Net is a second model only if receptive field limits the residual CNN. One-step diffusion is not justified for the first 8 GB experiment and would not be a faithful DLSS 5 replica.

Dataset records: `scene_id`, `trajectory_id`, `frame_index`, transforms, projection, seed, cheap/reference AO, depth, normal, cheap/reference RGB, validity mask, renderer commit/config. Save buffers in lossless formats with explicit units and color space. Split by **scene and trajectory**, never by randomly selected adjacent frames. Start with a few small scenes containing thin geometry, walls/corners, moving objects, and unseen held-out layouts. Automated capture is plausible through scripted camera motion and [Godot's fixed-FPS movie workflow](https://docs.godotengine.org/en/stable/tutorials/animation/creating_movies.html), but exact multi-buffer export must be implemented and verified.

## Temporal extension

Start stateless. Then warp previous AO using velocity, reject history at disocclusions using depth/normal checks, and add a small history feature or blend. Compare warped temporal error only on valid pixels, plus full-frame flicker and ghosting near newly revealed geometry. Reset state on camera cuts and resolution changes. [NRD](https://github.com/NVIDIA-RTX/NRD) and [SVGF](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/) are reference designs, not drop-in components for this Godot project.

## Execution on Apple Silicon

Start with offline capture and [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) training; fall back to CPU if a required operation fails. Benchmark model-only MPS separately from file I/O and conversions. Next test an external live loop only to quantify readback/synchronization cost; it is not the desired final architecture. For eventual engine integration, export PyTorch to a Core ML `mlprogram` via [coremltools](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html) with fixed [input shapes/types](https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html), and compare output numerically. Core ML accepts [CVPixelBuffer image inputs](https://developer.apple.com/documentation/coreml/mlfeaturevalue/init%28pixelbuffer%3A%29) and [CVMetalTextureCache](https://developer.apple.com/documentation/CoreVideo/cvmetaltexturecache-q3j) can share Core Video image buffers with Metal, but a direct Godot texture → Core ML image path needs a native extension and explicit synchronization. Apple's [Metal readback example](https://developer.apple.com/documentation/metal/reading-pixel-data-from-a-drawable-texture) shows why an offline CPU bridge can be costly. Zero copy is a hypothesis to test, not a baseline assumption. ONNX adds no clear value to the first macOS path; revisit only if Core ML conversion fails. Metal 4 ML passes exist but are not assumed available or useful on this M2 without OS/device validation.

The renderer and model share GPU and 8 GB unified memory. Record texture allocation, data conversion, command completion, inference, composite and present separately with [Instruments Game Performance/Metal trace](https://developer.apple.com/documentation/Xcode/Analyzing-the-performance-of-your-Metal-app). Profile thermal state and repeat warm runs. An inference duration alone cannot support a real-time claim.

## Evaluation contract

Compare four paths: cheap AO alone; conventional AO/filter tuned to a comparable **total millisecond** budget; cheap AO + neural model; expensive reference AO. Report AO and final-RGB PSNR/SSIM, edge error, optional LPIPS, temporal warped error and ghosting/disocclusion examples. Report average/p50/p95 end-to-end latency and FPS, render/AO/capture/convert/infer/composite/present times, CPU/GPU utilization and peak memory. Include model-only latency as a separate row. The result is useful only if held-out quality improves, geometry remains stable, and the neural path has an acceptable quality/cost trade-off against the matched conventional path. Do not claim speedup merely by beating cheap AO in quality.
