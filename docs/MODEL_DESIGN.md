# Model design and tensor operations

Decision: compare a deterministic RGB residual CNN, a small multiscale encoder-decoder, and a conditional diffusion teacher distilled to one call. On the original 96-pair RGB dataset, none beat the cheap input on test MAE. Later geometry-conditioned models improved aggregate test metrics but still failed on some scenes. The current focused task is to refine an existing low-resolution shadow cue. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Model A: implemented residual CNN

In [models/baseline.py](../models/baseline.py), input `x` is display RGB float in `[0,1]`, shape `B×3×H×W`. At initial validation, `H=W=128`; renderer resolution is `H=144,W=256`. `Conv2d(3,16,3,pad=1)` produces `B×16×H×W`; ReLU keeps nonnegative features. Two residual blocks each compute `h + Conv3×3(ReLU(Conv3×3(h)))`, preserving a direct feature path. `Conv2d(16,3,3,pad=1)` produces a learned RGB correction. Output is

`y_hat = clamp(x + δ tanh(r_theta(x)), 0, 1)`, with configurable `δ`.

The last convolution is zero initialized, so an untrained model starts as identity. The default `δ=0.25` was raised to `0.6` for the shadow experiment because 10.7% of target color-channel changes exceed 0.25. The bound does **not** guarantee object identity. The model has **10,163 trainable parameters** and a maximum local receptive field of 13×13 pixels. Removing the residual addition forces full RGB reconstruction and risks drift; removing blocks reduces contextual capacity; removing the bound permits larger appearance changes. The code's three-channel input means depth and normals are not yet consumed.

Each ordinary 3×3 convolution has `9 C_in C_out` weights and approximately `H W 9 C_in C_out` multiply-accumulates. One `B×16×144×256` activation is about 2.25 MiB in FP32 (half in FP16); training retains multiple activations and gradients. Parameters alone do not predict latency. The measured M2 model-only timing is in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Summing the six convolutions gives about **10,080 MACs per output pixel**, or **0.372 billion MACs** at 256×144 for batch 1. This arithmetic estimate omits ReLU, `tanh`, clamp, memory traffic, and framework overhead. The saved FP32 checkpoint is about 45 KiB; runtime activation and allocator memory are larger.

Example operations:

```python
model = ResidualRenderer(channels=16, blocks=2)
x = torch.rand(1, 3, 144, 256)
y = model(x)                  # (1, 3, 144, 256)
loss = (y - reference).abs().mean()
loss.backward()              # gradients through convolutions and residual path
```

RGB residual is the best first representation because existing Godot RGB capture works and it can learn bounded appearance changes without demanding material decomposition. Full RGB output has more freedom and greater structure risk. A lighting/material residual would be better constrained, but needs paired separated lighting/material targets. A latent generator adds a decoder and spatial alignment problem. A [U-Net](https://arxiv.org/abs/1505.04597) can add multiscale context with skips, but its extra activations and quality benefit must be measured. [MobileNetV2](https://arxiv.org/abs/1801.04381) motivates depthwise convolutions; fewer arithmetic operations need not mean lower MPS latency because operator launches and memory movement matter.

## Model B: implemented multiscale encoder-decoder

The implemented [MultiScaleRenderer](../models/multiscale.py) accepts `B×3×144×256` RGB. Its encoder produces 12 channels at 144×256, 16 at 72×128, 24 at 36×64, and 32 at 18×32. Bilinear upsampling and skip concatenation restore full resolution. A 3-channel head predicts a bounded RGB residual. It has 79,179 parameters. It was trained with the same loss and split as Model A. It stayed close to identity on the held-out scene, so larger receptive field alone did not solve shadow placement.

The same backbone can now accept `B×11×144×256` conditioning: 3 RGB channels, 1 depth channel (linear meters divided by 10 and clipped), 3 decoded view-normal channels, 1 valid-depth mask, and 3 spatially constant view-space light-direction channels. The first three channels are also the skip-connected RGB output base. This variant has 80,043 parameters. It used the same 20 epochs, weighted L1 loss, validation selection, and 16-layout split as the matched RGB-only backbone. Its changed-pixel test error fell from 0.3075 for the cheap input to 0.1239, but its unchanged-pixel error rose to 0.0261. This is a useful but imperfect shadow predictor; it worsened one of the two held-out scene layouts.

## Current masked shadow-refinement model

[`ShadowRefinementRenderer`](../models/multiscale.py) uses the same 11-channel input and multiscale features as Model B. A 1-channel `mask_head` predicts logits `m`; a 3-channel `strength_head` predicts signed RGB values `s`. The output is `clamp(cheap_RGB + 0.6 sigmoid(m) tanh(s), 0, 1)`. The mask has shape `B×1×H×W` and broadcasts over the three color channels. The correction has shape `B×3×H×W`. Both heads are learned 3×3 convolutions; the strength head is zero initialized so the network starts with unchanged RGB. The model has **80,152 parameters**.

The mask is supervised using pixels where cheap and reference differ by more than 5/255 in any channel. Training also uses L1 loss with weight `1+20×` on these pixels. This is ordinary deterministic regression. The mask has no hard guarantee against false positives, and the higher changed-pixel weight encourages changing some pixels that should stay alone. On four held-out scenes, the model improved aggregate MAE and PSNR but worsened one entire low-change scene. Its unchanged-pixel MAE rose from 0.000645 for cheap RGB to 0.004890. The false correction is visible on the floor in the saved comparison image. We cannot call this robust cross-scene rendering or a cost-effective game effect yet.

The choice of an existing low-resolution shadow cue narrows the inference problem: the network refines plausible shadow edges rather than deriving unseen occlusion from a single depth/normal view. It also means the cheap renderer already pays for a shadow pass. The practical test must compare `512 atlas + model` against a conventional atlas size of similar total render cost, including capture and handoff.

After fitting, one scalar mask-logit bias was selected on validation scenes. The deployed expression is `clamp(cheap_RGB + 0.6 sigmoid(m−1.5) tanh(s), 0, 1)`. No parameters were retrained. On four new procedural scenes it reduced MAE relative to cheap RGB in 47/48 frames, but the unchanged-pixel error remained above cheap RGB. Offline depth-reprojected temporal error was also worse than cheap RGB. A simple history blend helped that error slightly while increasing image MAE. Thus the current model is a useful spatial baseline with an unresolved temporal and end-to-end cost problem.

The [live M2 bridge](../evaluation/serve_live.py) now measures that cost directly for a serial external route. Its Godot renderer exported geometry and RGB, the PyTorch MPS model returned RGB over local TCP, and Godot displayed it. The steady pipeline took 55.17 ms median, far above a 30 FPS frame budget. A Metal trace showed the 512-to-4096 atlas change costs only about 0.2 ms of GPU work on the test scene. The current model is therefore an educational quality baseline, not a cost-effective substitute for this conventional shadow setting. More shadow training alone cannot fix that cost mismatch.

A controlled loss-weight comparison kept this architecture fixed and made changed and unchanged pixel weights explicit. `changed_weight=5, unchanged_weight=4` collapsed close to identity. The milder `5,2` setting improved held-out static and moving-scene MAE. The selected inference path adds a hard gate at the mask-logit bias: below threshold, it preserves input RGB exactly. Validation selected bias 1.0 under a 0.5% limit on unchanged pixels darkened by more than 5/255. The measured false-darkening rate was 0.347% on new static scenes and 0.577% on moving scenes. This reduces false darkening but does not eliminate it. Temporal history reduces error further, though the cheap renderer remains steadier. The temporal probe uses known object transforms offline, not engine motion vectors. Full numbers are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Model C: implemented diffusion-distillation experiment

The implemented [ConditionalNet](../models/one_step.py) uses a noisy shadow residual `x_t`, cheap RGB condition `c`, and normalized time `t`. The teacher has 30,979 parameters and predicts Gaussian noise. For target residual `r=y-c`, its DDPM-style training input is `x_t = √ᾱ_t r + √(1-ᾱ_t) ε` and loss is `||ε_θ(x_t,c,t)-ε||²`. An eight-step deterministic DDIM path generates a teacher residual from noise. The 30,835-parameter student receives the same initial noise and cheap RGB, then learns `||0.6 tanh(s_φ(z,c))-r_teacher(z,c)||²`. At inference it predicts an RGB residual in **one network call**. This is genuine distillation from a trained diffusion teacher, though the brief run produced poor images. The exact objective and architecture are our choices, not NVIDIA's.

The teacher and student operate in pixel space and use no VAE. At 256×144, adding a VAE would introduce encoder/decoder cost and reconstruction errors before we have evidence that pixel-space memory is the bottleneck. A one-call student is not guaranteed to be fast enough for a complete game pipeline; only model-call latency was measured. Fixed per-frame noise seeds made the still-image comparison repeatable, but temporal consistency has not been tested.

### Alternative research route, not implemented

One alternative route is conditional flow matching: sample target `y`, condition `c`, noise `z~N(0,I)`, and `t~U[0,1]`; set `x_t=(1-t)z+t y` and train `v_theta(x_t,t,c)` against `y-z` with squared error. A one-call sample `z+v_theta(z,0,c)` is a single Euler approximation to a generative flow. This is **flow-based**, not diffusion, and one step may be inaccurate. Merely adding a noise input to Model A or running a U-Net once does not suffice.

Illustrative flow-training operation (future design, **not executed**):

```python
z = torch.randn_like(target)
t = torch.rand(target.shape[0], 1, 1, 1, device=target.device)
x_t = (1 - t) * z + t * target
velocity = network(x_t, t, condition_rgb)  # network must accept time and condition
loss = ((velocity - (target - z)) ** 2).mean()
one_step = z + network(z, torch.zeros_like(t), condition_rgb)
```

Temporal conditioning will first warp the previous enhanced frame with engine motion vectors, reject disoccluded pixels using depth/normal tests, and feed valid history as an optional input. This can reduce flicker but can also create ghosting; it must beat a no-history baseline on held-out sequences.
