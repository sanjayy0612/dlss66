# DLSS 5: published architecture and open questions

Research checked 2026-11-12. NVIDIA's [research page](https://research.nvidia.com/labs/adlr/DLSS5/) (2026-09-01), [developer article](https://developer.nvidia.com/blog/whats-new-for-game-developers-dlss-5-with-3d-guided-neural-rendering-nvidia-ace-updates-and-new-rtx-kit-capabilities/) (2026-09-22), and [GeForce explanation](https://www.nvidia.com/en-gb/geforce/news/dlss-5-3d-guided-neural-rendering/) (2026-09-01) are the primary public sources. The linked research PDF could not be fetched here (HTTP 403), so claims below use those accessible pages.

## Confirmed NVIDIA descriptions

- DLSS 5 is a final-stage, one-step **pixel-space diffusion** appearance renderer. NVIDIA describes output as causal, deterministic, and temporally stable, up to 4K on RTX 50 hardware. These are vendor claims, not measurements in this repository.
- At inference, NVIDIA names the current rendered color frame, engine motion vectors, carried temporal state, and artistic control values. It describes structure and tone intensity, model selection, and semantic/engine masks. Renderer-derived attributes such as albedo, lighting, and normals guide consistency during **training**; public material does not establish normals as mandatory inference inputs.
- It changes appearance such as lighting and materials. Super Resolution reconstructs resolution, Frame Generation adds temporal samples, and Ray Reconstruction reconstructs ray-traced signals. Published FPS examples combine features and do not isolate DLSS 5 cost.

## What the sources do not disclose

The exact layers, width, parameter count, noise schedule, diffusion/distillation objective, data corpus, temporal-state representation, structure-loss formula, output parameterization, training compute, and isolated per-frame latency are unknown from the accessible pages. Our model choices below are independent proposals, not reconstructions of NVIDIA internals.

## Diffusion mathematics relevant to our research

The [DDPM paper](https://arxiv.org/abs/2006.11239) corrupts a target image `y` with Gaussian noise: `x_t = sqrt(alpha_bar_t) y + sqrt(1-alpha_bar_t) epsilon`, with `epsilon ~ N(0,I)`. A conditional noise predictor can train with `E ||epsilon_theta(x_t,t,c)-epsilon||²`, where `c` is renderer conditioning. Standard sampling repeatedly moves through noise levels; one call to an RGB regressor trained with L1 is **not** diffusion.

[Progressive distillation](https://arxiv.org/abs/2202.00512) trains a student to match multiple teacher sampling steps and repeatedly reduces the step count. [Consistency models](https://arxiv.org/abs/2303.01469) learn a mapping approximately constant along a probability-flow trajectory, with a clean-data boundary condition; they support one-step output. [Improved direct consistency training](https://arxiv.org/abs/2310.14189) reduces teacher dependence but still needs careful trajectory and schedule design. [Flow matching](https://arxiv.org/abs/2210.02747) learns a continuous transport velocity; one Euler step is only an approximation unless the path is sufficiently straight or distilled, as [InstaFlow](https://arxiv.org/abs/2309.06380) investigates.

For a later conditional consistency experiment, a schematic loss is `||f_theta(x_t,t,c) - stopgrad(f_ema(x_s,s,c))||²` for points on the same trajectory. The boundary condition and trajectory construction matter. This equation describes a research family, not DLSS 5's undisclosed training loss.

Pixel-space output avoids a learned decoder and can preserve exact pixel registration with the renderer, but feature tensors scale with `H×W`. A single step reduces repeated network calls yet may be hard to train, may hallucinate detail, and can still be too slow when graphics and inference share a GPU. Conditioning on RGB, motion, and state constrains output; it does not mathematically guarantee geometry fidelity. Masks, edge/identity losses, and held-out evaluation are needed.

Determinism can use a fixed noise field derived from frame/scene IDs or a model that maps a fixed latent plus conditioning to output. A changing independent noise seed per frame risks flicker. We will test deterministic baselines before temporal or generative claims.
