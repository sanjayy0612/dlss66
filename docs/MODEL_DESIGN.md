# Model design and tensor operations

Decision: begin with a deterministic RGB residual CNN, then test a conditional encoder-decoder, and attempt a genuine one-step generative model only when paired renderer data and a suitable training objective exist. The current code is Model A. This prioritizes a measurable baseline over a premature diffusion label.

## Model A: implemented residual CNN

In [models/baseline.py](../models/baseline.py), input `x` is display RGB float in `[0,1]`, shape `B×3×H×W`. At initial validation, `H=W=128`; renderer resolution is `H=144,W=256`. `Conv2d(3,16,3,pad=1)` produces `B×16×H×W`; ReLU keeps nonnegative features. Two residual blocks each compute `h + Conv3×3(ReLU(Conv3×3(h)))`, preserving a direct feature path. `Conv2d(16,3,3,pad=1)` produces a learned RGB correction. Output is

`y_hat = clamp(x + 0.25 tanh(r_theta(x)), 0, 1)`.

The last convolution is zero initialized, so an untrained model starts as identity. The 0.25 bound limits per-channel changes; it does **not** guarantee object identity. The model has **10,163 trainable parameters** and a maximum local receptive field of 13×13 pixels. Removing the residual addition forces full RGB reconstruction and risks drift; removing blocks reduces contextual capacity; removing the bound permits larger appearance changes. The code's three-channel input means depth and normals are not yet consumed.

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

## Model B: proposed conditional encoder-decoder

Input: RGB plus optional linear depth, view normals, and masks (`B×(3+1+3+1)×H×W`). A stride-2 encoder maps to `B×C×H/2×W/2`; a deeper block gathers wider context; upsampling and skip concatenation restore `H×W`; a 3-channel head predicts a bounded residual. All encoder/decoder weights are learned. Its larger receptive field may help broad lighting changes, while skips retain boundaries. It has more activation memory and may blur fine detail. We will compare it to Model A on the **same** paired renderer split and cost budget.

## Model C: later genuine one-step generator

One practical research route is conditional flow matching: sample target `y`, condition `c`, noise `z~N(0,I)`, and `t~U[0,1]`; set `x_t=(1-t)z+t y` and train `v_theta(x_t,t,c)` against `y-z` with squared error. A one-call sample `z+v_theta(z,0,c)` is a single Euler approximation to a generative flow. This is **flow-based**, not diffusion, and one step may be inaccurate. A diffusion-family one-step claim would require consistency training or distillation with noise/time conditioning and the corresponding trajectory objective. Merely adding a noise input to Model A or running a U-Net once does not suffice. No Model C code or training is implemented yet.

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
