# Paired shadow-rendering datasets

## Current experiments

The main task now asks whether a small network can **refine an existing cheap shadow**, rather than invent a shadow from shadow-free RGB. In `shadow_resolution` mode, [`dataset.gd`](../dataset.gd) renders the same seeded scene and camera twice with directional shadows enabled: 512-pixel atlas for the cheap pass and 4096-pixel atlas for the reference. The atlas size is set with Godot 4.7.2 `RenderingServer.directional_shadow_atlas_set_size(size, true)`. This gives a useful low-quality shadow cue to the network. It also makes the problem narrower than DLSS 5. A higher atlas size is a conventional graphics reference, not physical ground truth.

```sh
DLSS66_OUTPUT=dataset/shadow_resolution DLSS66_TASK=shadow_resolution DLSS66_LAYOUTS=32 sh collect_dataset.sh
DLSS66_OUTPUT=dataset/shadow_resolution python3 validate_dataset.py
```

The executed M2 capture has 32 layouts × 12 camera poses = 384 pairs. Scenes 0–23 train, 24–27 validate, and 28–31 test. The validator reported 288/48/48 pairs, mean cheap/reference difference 2.620/255, 9.17% of pixels changed by more than five code values, view-depth range 0.765–9.332 m, maximum unit-normal error 1.2×10⁻⁷, and no sky-mask disagreement. The generated files are local and ignored by Git. The test scene IDs were held out from fitting and checkpoint choice, but later diagnostic inspection of them means further tuning requires a new untouched test split.

New scene IDs 32–35 were captured after validation-only mask calibration into `dataset/shadow_resolution_fresh/` with `DLSS66_SCENE_START=32 DLSS66_LAYOUTS=4 DLSS66_OUTPUT=dataset/shadow_resolution_fresh DLSS66_TASK=shadow_resolution sh collect_dataset.sh`. All 48 pairs are marked `test`; the directory is ignored by Git. This is a fresh in-distribution check on the same procedural scene generator, not a new art style or engine.

The original `shadow_off` mode is still supported by `sh collect_dataset.sh`. The latest local copy has 40 layouts × 12 poses = 480 pairs: 336 train, 24 validation, 24 development, 96 fresh test. Its latest validation reported mean cheap/reference difference 9.482/255. Older 16-layout results below are retained as historical experiments.

## Purpose and generation

The earlier dataset tested whether renderer geometry helps a small model add directional shadows. [`dataset.gd`](../dataset.gd) creates seeded layouts with boxes, cylinders, spheres, varied colors, object counts, floor colors, and light directions. Each layout has 12 camera poses. The cheap pass disables directional shadows; the reference pass enables them. Both passes rebuild the same seeded scene and use the same camera pose. The reference is a Godot rendering choice, not physical ground truth or a DLSS 5 output.

The cheap pass attaches [`dataset_probe.gd`](../dataset_probe.gd), which uses the same proven [`export.glsl`](../export.glsl) compute shader as the Stage 1 probe. At `POST_OPAQUE`, it samples Godot's resolved depth and normal/roughness textures, decodes the view-space normal, writes depth and normal into an application-owned RGBA32F texture, and requests asynchronous readback. The viewport RGB is saved after geometry readback while the scene and camera remain static. The reference process uses the same procedural scene and camera without the probe. Both use `Environment.ssao_enabled=true` with intensity zero to make the normal/roughness texture available; this may add renderer overhead and is not a free inference buffer.

Run the default shadow-off task on the pinned official Godot 4.7.2 stable app in `tools/` with NumPy and Pillow installed:

```sh
sh collect_dataset.sh
```

For a four-pair pilot: `DLSS66_LAYOUTS=2 DLSS66_FRAMES=2 sh collect_dataset.sh`. The output is local in Git-ignored `dataset/raw/`; `.gdignore` prevents Godot from importing generated images. No game assets or external dataset are downloaded.

## Files and conditioning

The paired `manifest.jsonl` has relative paths to `cheap`, `reference`, and `geometry`, plus scene and frame IDs, split, seed, camera pose, near/far, and view-space light direction. Each `geometry` file is little-endian float32 with shape `144×256×4`: reverse-Z device depth in channel 0 and decoded view-space normal in channels 1–3. Background depth and normal are zero. For valid pixels, positive view distance in meters is `near × far / (near + depth × (far - near))`.

[`training/paired.py`](../training/paired.py) turns the file into 11 model channels: cheap RGB (3), metric depth divided by 10 and clipped (1), view normals (3), valid-depth mask (1), and view-space light direction repeated across pixels (3). The divisor 10 is a dataset-scale choice; a broader scene range needs a better depth normalization. Geometry is from the cheap pass only. The reference pass differs in shadows, while scene seed, camera pose, and light direction match in the manifest.

## Earlier 16-layout validation on Apple M2

| Property | Observed |
|---|---:|
| Resolution | 256×144 |
| Scenes × camera poses | 16 × 12 |
| Train / validation / test | 144 / 24 / 24 pairs, split by scene |
| Cheap / reference PNGs | 192 / 192 |
| Geometry exports | 192, each 589,824 bytes |
| Dataset directory size | ~112 MiB on disk |
| Distinct geometry SHA-256 hashes | 192 |
| Distinct light orientations | 16 |
| Mean cheap/reference RGB absolute difference | 12.508 / 255 |
| Minimum / maximum pair difference | 0.818 / 33.965 / 255 |
| Valid-depth fraction, mean | 79.45% |
| Positive view-depth range | 0.800–9.332 m |
| Maximum foreground normal-length error | 1.2×10⁻⁷ |
| Cheap/reference sky-mask disagreement | 0% |

[`validate_dataset.py`](../validate_dataset.py) checks pair counts, unique keys, matching scene/camera/light metadata, image dimensions, exact geometry byte count, finite depth/normals, valid depth range, unit foreground normals, zero background normals, and sky-mask registration against depth and both RGB images. Some views are fully occupied by geometry, so they correctly have no sky pixels. `sample_depth.png` and `sample_normal.png` in `dataset/raw/` were visually checked against the first cheap RGB frame; objects, floor, walls, and background align. Sky-mask equality and visual inspection do not prove every interior object edge is aligned to subpixel precision.

## Model result and limits

On this 16-layout split, the matched RGB-only multiscale model had test MAE 0.04247, while the cheap input had 0.03978. The geometry-conditioned version reached 0.03878 and improved test PSNR from 18.67 to 20.76 dB. It improved only 10 of 24 test frames: scene 14 worsened, while scene 15 improved strongly. It also introduced dark halos around some objects. Full measurements and commands are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). The model does not yet show robust cross-scene shadow prediction.

There are still only 16 simple procedural layouts, one static light per layout, and correlated views within each scene. No moving objects, motion vectors, temporal state, or VAE are included. Model-call timing excludes Godot rendering and geometry readback. A previous 8-layout, 96-pair RGB-only dataset and its model results are recorded in the repository at commit `778cc70`; the current capture overwrites local manifests with the expanded version.
