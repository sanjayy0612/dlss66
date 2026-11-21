# Temporal shadow-refinement architecture

This page describes the implemented `ConfidenceGatedTemporalRenderer`. It is a small research model for refining an existing low-resolution shadow cue. It is not NVIDIA DLSS and is not connected to the live Godot renderer as an end-to-end inference path.

## Per-frame inputs

The model receives 15 full-resolution channels at 256×144:

| Channels | Values |
|---:|---|
| 3 | Current cheap-render RGB |
| 1 | Linear view depth, normalized by 10 m and clipped to `[0,1]` |
| 3 | View-space surface normal |
| 1 | Valid-depth mask |
| 3 | View-space light direction, repeated over the image |
| 3 | Previous predicted RGB correction, reprojected into the current view |
| 1 | Reprojection correspondence-valid mask |

The previous correction is depth-tested and warped from the previous frame. A missing or invalid correspondence has zero history weight.

## Network and output

The README includes a [concept illustration](../README.md#how-the-shadow-refinement-experiment-works). It is illustrative, not a screenshot or a literal network schematic. Exact architecture details follow.

The encoder reduces spatial resolution in three steps. Its feature widths are 12 at full resolution, 16 at half resolution, 24 at quarter resolution, and 32 at one-eighth resolution. The middle block keeps 32 channels. The decoder upsamples and concatenates the matching encoder features, then returns to 24, 16, and 12 channels at full resolution. Each encoder and decoder block uses two 3×3 convolutions with ReLU.

The network has 80,693 parameters.

The mask head predicts where a correction is allowed. The strength head predicts a signed RGB residual bounded by `max_delta=0.6`. The confidence head predicts how much reprojected history to use. At inference, history fusion is further limited by geometric correspondence and the current hard correction-mask support:

```text
history_weight = sigmoid(confidence_logit) × correspondence_valid × current_mask_support
output = clamp(current_rgb + (1 − history_weight) × current_correction
                + history_weight × previous_correction, 0, 1)
```

## Training and evaluation path

The training reference uses the same scene and camera as the cheap input, with a higher-resolution directional-shadow atlas. Training combines weighted image error, a correction-mask loss, temporal residual consistency across three-frame rollouts, confidence regression, and a penalty for darkening unchanged pixels. A boundary-weighted version of the preservation penalty was also tested; it did not pass the held-out preservation target.

The model is evaluated by frame MAE, reprojected temporal error, and the rate of darkening on pixels whose reference changed by no more than 5/255. Current results show frame-error reduction on these procedural scenes, while temporal error remains above the cheap renderer and preservation varies across held-out scenes. See [the experiment log](EXPERIMENT_LOG.md) and [the final findings](TEMPORAL_NEXT_STEP_RESEARCH.md).
