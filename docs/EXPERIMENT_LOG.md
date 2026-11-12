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

## Paired renderer capture, executed 2026-11-12

`sh collect_dataset.sh` ran Godot 4.7.2 on the M2 Metal Forward+ renderer. It captured 96 pairs of 256×144 RGB PNGs from eight procedural scene layouts and 12 camera poses per layout. The two passes differed by directional shadows off versus on. `validate_dataset.py` checked matching seed, camera pose, dimensions, file presence, unique keys, and nonzero image differences. The scene split is 72 train, 12 validation, and 12 test pairs. Mean absolute cheap/reference difference was 10.199/255 per channel; the smallest pair was 2.041/255 and the largest 18.297/255. On average, 12.31% of pixels differed by more than five levels in at least one channel. A four-pair pilot produced identical SHA-256 PNG hashes on two runs. One sampled PNG pair and the manifest had matching hashes across two full runs. Full-corpus hash reproducibility and geometry-mask alignment were not independently measured. Details: [DATASET.md](DATASET.md).

## Planned, not executed

- Repeat full-corpus hashes and verify object-mask alignment if this dataset is used for a quantitative training claim.
- Train Model A on those pairs with held-out scenes; compare against input and conventional appearance baselines at matched total cost.
- Add a compact encoder-decoder if broader context improves held-out quality enough to justify extra memory/latency.
- Build a conditional flow or consistency objective with noise/time input. Quantify teacher or trajectory cost before claiming one-step diffusion-family generation.
