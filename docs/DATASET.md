# First paired renderer dataset

## Task and source

This local dataset tests whether a compact RGB-conditioned model can add shadows to a cheap render. `dataset.gd` builds each layout from seeded procedural boxes, floor, walls, and one directional light. The cheap pass disables the light's shadow map; the reference pass enables it. Both use the same camera pose and scene seed. The reference is a conventional Godot render, not ground-truth global illumination or a DLSS 5 target. Shadows require geometric knowledge that RGB may not fully reveal, so this is a useful test of the RGB-only model's limits.

Godot 4.7.2 stable on the Apple M2, Metal Forward+, captured both passes in separate processes. This avoids an intermittent stall observed when toggling shadows within a live scene. Code: [`dataset.gd`](../dataset.gd), [`dataset.tscn`](../dataset.tscn), [`main.gd`](../main.gd), [`collect_dataset.sh`](../collect_dataset.sh), [`validate_dataset.py`](../validate_dataset.py).

## Reproduce

Place the pinned `Godot.app` in `tools/`, install Python NumPy and Pillow if needed, then run from the repository root:

```sh
sh collect_dataset.sh
```

For a small pilot, use `DLSS66_LAYOUTS=2 DLSS66_FRAMES=2 sh collect_dataset.sh`. The script first imports the Godot project, then captures cheap and reference passes, and finally validates and pairs them. Output is local and Git-ignored under `dataset/raw/`. `.gdignore` prevents Godot from importing generated PNGs back into the project.

The paired `dataset/raw/manifest.jsonl` has relative paths, scene ID, frame ID, split, seed, camera position, look-at target, effect, and resolution. The files `manifest_cheap.jsonl` and `manifest_reference.jsonl` preserve each pass's capture metadata. `summary.json` records validation statistics. The source code and seed regenerate the data; no game or copyrighted assets are downloaded.

## Executed result

| Property | Observed |
|---|---:|
| Resolution | 256×144 RGB |
| Layouts and camera poses | 8 layouts × 12 poses |
| Train / validation / test | 72 / 12 / 12 pairs, split by layout |
| PNG files | 192 |
| Dataset directory size | 1.7 MB on disk |
| Mean absolute cheap/reference RGB difference | 10.199 / 255 |
| Minimum / maximum pair difference | 2.041 / 18.297 / 255 |
| Mean fraction of pixels changing by >5 in any channel | 12.31% |

The validator checks that every pair exists, has 256×144 RGB shape, has matching scene seed and camera metadata, and has a nonzero difference. A four-pair pilot had identical SHA-256 hashes on two runs. After a second complete 96-pair run, the hash of its paired manifest and one sampled cheap/reference PNG pair also matched the earlier full run. The reference images were visually inspected: dark changes occur where objects cast shadows; objects retain their apparent positions. Full-corpus hash identity and pixel-level object-mask alignment remain unverified.

## Limits and next use

The scenes contain boxes with simple materials, only eight layouts, and a single lighting change. Camera poses within each layout are correlated. This is enough to test data loading and a first held-out-scene model comparison; it cannot establish broad scene generalization or DLSS-like generative appearance. The dataset contains RGB only. Prior depth/normal capture is a separate experiment and has not been aligned with these 96 pairs. Rendering time, shadow-map cost, and end-to-end model inference cost have not been measured here.

One complete rerun stalled during Godot rendering in the restricted command sandbox and produced incomplete manifests. The validator rejected those partial files. A subsequent run with normal GUI access completed in about 11 seconds. Capture timing therefore varies by execution environment; 11 seconds is not a rendering benchmark.

The first CNN, multiscale, and diffusion-distillation experiments on this split are recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). None improved whole-image test MAE over the cheap input. Add aligned geometry guidance and more varied scenes before making a broader model-quality claim; compare against a conventional shadow or screen-space baseline before making a performance claim.
