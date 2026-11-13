# Training and evaluation methodology

## Data ladder

1. **Executed smoke task:** seeded synthetic colored rectangles with a deterministic spatial illumination transform. It verifies tensors, optimization, checkpointing, and MPS operation. It says nothing about realistic neural rendering or scene generalization.
2. **Executed paired task:** 96 Godot shadow-off/shadow-on pairs at identical resolution and camera pose. The split uses six training layouts, one validation layout, and one test layout. See [DATASET.md](DATASET.md). Depth and normals are not aligned to these pairs.
3. **Later:** add moving objects, motion vectors, temporal state, and controlled disocclusion/camera-cut sequences. Reference output is a chosen renderer configuration, not necessarily physical ground truth.

For paired deterministic training, the initial objective is `L_rec = mean |y_hat-y|`. Optional edge fidelity is `L_edge = mean |∇y_hat-∇y|`; optional perceptual fidelity is `L_perc = ||φ(y_hat)-φ(y)||`, with a fixed feature extractor. These are **our proposed losses**, not NVIDIA's undisclosed DLSS 5 losses. A bounded residual and input-consistency masks can protect structures, but must be checked on object boundaries. [Deep Shading](https://arxiv.org/abs/1603.06078) establishes geometry-conditioned learned shading as related prior work.

For a genuine diffusion-family model, use noise/time conditioning plus a [DDPM](https://arxiv.org/abs/2006.11239) noise objective, then quantify the cost of [distillation](https://arxiv.org/abs/2202.00512) or [consistency training](https://arxiv.org/abs/2303.01469) to reach one step. The current experiment trains an eight-step conditional diffusion teacher and distills its deterministic DDIM outputs into a one-call student. Its poor output is recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Conditional flow matching uses `E||v_theta((1-t)z+ty,t,c)-(y-z)||²` and a one-step Euler approximation; that is a separate generative method. A regression loss alone cannot support a diffusion claim.

## Evaluation protocol

Report PSNR and [SSIM](https://doi.org/10.1109/TIP.2003.819861) on held-out paired RGB; add [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) if dependency and runtime are justified. Inspect input, target, output and absolute-error maps. A perceptual gain can come with pixel or identity loss; report both, consistent with the [perception-distortion tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html). Check edge displacement, object colors and texture invention by region, not only a global scalar.

For temporal evaluation, warp previous output with engine motion vectors `W(y_hat_{t-1},v_t)` and measure `mean M_t |y_hat_t-W(y_hat_{t-1},v_t)|` on valid, non-disoccluded pixels. Compare with the same warp error on reference frames; a constant/oversmoothed video can otherwise appear stable. Inspect newly revealed boundaries for ghosting. [FlowVid](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_FlowVid_Taming_Imperfect_Optical_Flows_for_Consistent_Video-to-Video_Synthesis_CVPR_2024_paper.html) motivates flow-and-occlusion conditioning. Reset history on cuts.

Benchmark the exact same resolution and precision after warmup. Synchronize MPS/CUDA before stopping each timer. Report parameter count, checkpoint bytes, peak device/process memory, mean/median/p95 model-only latency, and render/copy/preprocess/inference/present end-to-end time separately. Consumer-hardware suitability is an experimental outcome, not a property inferred from a small parameter count.
