# Experiment log

## Executed 2026-11-12

Environment: MacBook Air M2, 8 GB unified memory; Python 3.13; PyTorch 2.13.0; native MPS. Existing Godot RGB/depth/normal capture remains available. The new model currently uses RGB only.

| Experiment | Command | Result |
|---|---|---|
| CPU shape, identity, gradient, synthetic determinism, checkpoint round-trip | `python3 -m unittest discover -s tests -v` | 3 tests passed in 0.981 s |
| Short MPS training smoke | `python3 -m training.train --steps 8 --height 128 --width 128 --device mps` | 10,163 parameters; initial train L1 0.09796; held-out input L1 0.09938; model L1 0.02902; ~45 KiB checkpoint saved |
| M2 model-only inference | `python3 -m evaluation.benchmark --device mps --height 144 --width 256 --warmup 10 --samples 30` | mean 1.520 ms; median 1.487 ms; p95 1.768 ms, with MPS synchronization each sample |

The synthetic target is a fixed lighting/color transform on procedurally generated rectangles. The training result proves the software path learns that simple mapping; it is **not evidence** of realistic appearance generation, scene transfer, temporal stability, or DLSS-like quality. The inference result excludes Godot rendering, frame readback, tensor creation, compositing, and presentation. No memory peak or full-frame FPS was measured. No generative model was trained.

Earlier renderer capture experiments and their limits are in [TEST_RESULTS.md](../TEST_RESULTS.md).

## Planned, not executed

- Capture paired cheap/reference Godot RGB from distinct scene layouts and camera trajectories; verify exact alignment and metadata.
- Train Model A on those pairs with held-out scenes; compare against input and conventional appearance baselines at matched total cost.
- Add a compact encoder-decoder if broader context improves held-out quality enough to justify extra memory/latency.
- Build a conditional flow or consistency objective with noise/time input. Quantify teacher or trajectory cost before claiming one-step diffusion-family generation.
