# Implementation-oriented research plan

Historical AO-centered plan (2026-11-11). See [MODEL_DESIGN.md](MODEL_DESIGN.md), [TRAINING_METHODOLOGY.md](TRAINING_METHODOLOGY.md), and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for the current ML-first direction.

Research snapshot: 2026-11-11. No implementation or training is authorized in this phase.

## Decision

Primary testbed: pinned Godot 4 Forward+ scene with MIT/open assets. Later game demonstration: SuperTuxKart, only after checking native build, legal assets, renderer hook and deterministic replay. Fallback: a minimal native Metal scene if Godot cannot provide dependable buffer capture or effect insertion. Classic Super Mario Bros. is 2D; SM64 native ports add asset and renderer work without helping the first scientific question.

**First independent experiment:** at 256×144, generate matched 4–8-sample and 32–64-sample screen-space AO for fixed camera paths. Train a small residual CNN on cheap AO + depth + normals to estimate high-sample AO. Evaluate on held-out scene layouts against cheap AO, a bilateral AO filter, and a conventional higher-sample setting at matched end-to-end cost. This teaches G-buffer use, supervised shading reconstruction, renderer coupling, and the gap between network latency and delivered frame time.

## Stages and gates

| Stage | Concrete milestone | Progress only when |
|---|---|---|
| 0 — Graphics | Read a Godot renderer path and make diagrams of vertex/index input, raster/depth, shading, compositor, present; inspect one GPU frame | Buffer names and pass timing are observed, not guessed |
| 1 — Testbed | Pin Godot version/demo commit; run on M2; script deterministic camera; visualize and export RGB, linear depth, view normals; log settings | Repeated same-state captures match within documented tolerance; normals/depth decode verified |
| 2 — Classical | Implement explicit AO pass with adjustable sample count and bilateral/filter baseline; measure per-pass time | Cheap/reference pair is aligned; quality/cost curves exist; target is acknowledged as screen-space approximation |
| 3 — Neural | Capture train/validation/test by scene and path; train ~10–20k-parameter CNN at 256×144; evaluate AO/RGB | Held-out quality beats cheap path, and comparison with matched conventional cost is reported even if neural loses |
| 4 — Temporal | Capture velocity; validate direction and disocclusion; add history only after stateless baseline | Flicker/ghosting metrics improve without major edge artifacts |
| 5 — Integration | Prototype Core ML export and native engine bridge; compare against offline output | Numerical agreement and full-frame p50/p95 latency measured; live result shown side by side |
| 6 — Optimize | Profile GPU/CPU/memory; try precision, channel count, buffer reuse or quantization one at a time | Each change has measured before/after quality and end-to-end time |

If Stage 1 buffer access fails in the pinned Godot release, investigate its extension API once, then switch to the custom Metal fallback. If a simple conventional AO setting wins at matched latency, report that result and use it to select a different effect (such as low-sample lighting denoising); do not force a neural success narrative. Use the RTX 4050 only after M2 baseline and reproducibility are established.

## Suggested repository layout for the later implementation

```text
godot/                 pinned project, scripts, AO/compositor shaders
capture/               export and dataset validation tools
model/                 baseline network and training
evaluate/              image, temporal and performance metrics
configs/               renderer/model settings and seeds
data/manifests/        small metadata only; generated frames ignored
results/               reproducible tables and selected visuals
docs/                  graphics path, decisions, runbook
```

Initial dependencies: Godot 4 stable Forward+; Python 3; PyTorch with MPS; NumPy; an image/array format reader; scikit-image for SSIM if needed. Pin versions only after Stage 1 compatibility checks. Core ML Tools and a native GDExtension are deferred to Stage 5. No large downloads or copyrighted game assets are needed for the first experiment.

## Minimum successful result

A scriptable paired capture from one 3D scene and held-out path; verified AO/depth/normal alignment; reproducible train/test split; a trained tiny CNN; quality and temporal plots/tables; measured render, transfer, model and total frame costs; comparison with a cost-matched conventional AO path. A negative result is still successful research if it establishes why the neural path is worse. Real-time integration is a later gate, not a prerequisite for this first result.
