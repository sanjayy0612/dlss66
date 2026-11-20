# Next step for temporal shadow refinement (2026-11-17)

## Decision

Fix the support of the existing confidence-gated history path before adding a larger network layer. A reprojected pixel can be geometrically valid while its old shadow correction is wrong for the current lighting. NVIDIA's RTXDI/NRD integration explicitly treats lighting-history confidence separately from reprojection: changed lighting should reset accumulated history, because stale history can make shadows lag or smear. [RTXDI confidence documentation](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Confidence.md). SVGF and temporal supersampling also validate history against current-frame information before accumulation. [SVGF paper](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/); [NVIDIA temporal supersampling presentation](https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf).

## Repository evidence

The verified MPS recurrent checkpoint improves frame MAE on held-out static and moving scenes, but its temporal error remains above cheap rendering: 0.005747 vs 0.005108 and 0.006646 vs 0.006195, respectively. False darkening is below the 0.1% cap. These values are in `artifacts/recurrent_mps_static_test.json` and `artifacts/recurrent_mps_motion_test.json` and documented in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

The new, uncommitted `ConfidenceGatedTemporalRenderer` trained for three epochs on MPS. Its selected epoch has lower validation image MAE than the prior recurrent model, but validation false darkening is 0.182% static and 0.211% moving, above the 0.1% cap; temporal error is still above cheap input (0.005820 vs 0.005209 static, 0.006388 vs 0.006042 moving). Source: `artifacts/shadow_refine_confidence_m2_mps.json`. It has not earned a held-out evaluation yet.

The direct code-level risk is in `models/temporal.py`: the current prediction is multiplied by the hard correction mask, but `fuse_history()` blends the previous correction whenever correspondence is valid. It can therefore produce a correction where the current hard mask rejects one. This is an observed code path and a plausible cause of the increased false darkening; its contribution has not yet been isolated experimentally.

## Smallest useful experiment

1. Make history fusion obey the **current correction support** as well as geometric correspondence. At the selected hard-mask bias, no history correction should pass through a pixel rejected by the current mask. Add a unit test for that invariant.
2. Keep the trained checkpoint fixed. Evaluate validation scenes with and without this support constraint, including frame MAE, temporal error, and false-darkening rate. Do not tune on held-out scenes.
3. If validation meets the 0.1% false-darkening cap and improves temporal error over the prior recurrent checkpoint, evaluate untouched static scenes 36–39 and moving scenes 40–41. Otherwise, examine current-frame shadow evidence and history disagreement before adding channels or network width. A geometry-valid mask alone does not encode a changed shadow. [RTXDI confidence documentation](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Confidence.md).

This is a hypothesis, not a claimed fix. Restricting support may sacrifice some frame MAE. The target is a safe improvement over the verified recurrent baseline, ideally temporal error no worse than cheap rendering, under the same 0.1% false-darkening cap.

## Integration order

Only after offline quality passes, profile the recurrent path inside Godot. The existing 3.80 ms median synthetic MPS path excludes render synchronization, readback, IPC, and presentation; the earlier geometry-conditioned Godot bridge measured 166.58 ms median. Godot's `CompositorEffect` exposes depth, normals, and motion vectors in its render callback and can request motion vectors, so an engine-side history path is technically plausible, but its actual buffer access and synchronization cost must be measured in the pinned Godot build. [Godot CompositorEffect reference](https://docs.godotengine.org/en/stable/classes/class_compositoreffect.html).

No evidence here justifies a transformer, diffusion stage, or wider model. The current failures concern temporal validity and the engine bridge, not demonstrated model capacity.

## Executed support-mask experiment

History fusion now uses the current hard correction mask as well as reprojection validity. The existing MPS confidence checkpoint was kept fixed. On moving validation at bias 4, false darkening fell from 0.211% to 0.180%, still over the 0.1% limit. Bias 5 was then selected using validation only: static and moving false darkening were 0.084% and 0.089%, with frame MAE 0.008282 and 0.008844, and temporal error 0.005843 and 0.006364. All four error values were slightly lower than the prior recurrent checkpoint's validation values, but temporal error remained higher than the cheap input.

The fixed bias-5 setting was evaluated once on held-out scenes. Static scenes 36–39 gave frame MAE 0.008095, temporal error 0.005481, and false darkening 0.039%. Moving scenes 40–41 gave frame MAE 0.009846, temporal error 0.006522, and false darkening **0.102%**, narrowly failing the 0.1% preservation limit. Cheap-renderer temporal errors were 0.005108 static and 0.006195 moving. Results are in ignored local artifacts `confidence_masked_static_test_bias5.json` and `confidence_masked_motion_test_bias5.json`.

The fixed checkpoint improves frame and temporal error relative to the prior recurrent model on these scenes, but does not meet the full acceptance criteria. Do not tune bias on this held-out set. The next training experiment should penalize false correction on unchanged pixels, select checkpoints under a validation preservation constraint, and then use a fresh held-out scene set. The existing confidence head is sufficient for that experiment; network width is not the demonstrated bottleneck.

## Preservation-training experiment

That training experiment was run from `shadow_refine_confidence_m2_mps.pt` for three MPS epochs. The new loss penalizes darkening beyond 5/255 only where cheap and reference RGB differ by at most 5/255. Checkpoint selection requires false darkening at most 0.1% on both validation sets at bias 5. All three epochs passed validation; epoch 3 had the best eligible objective. Its static and moving validation false-darkening rates were 0.063% and 0.065%. The model and run log are ignored local artifacts `shadow_refine_confidence_preserved_mps.pt` and `.json`.

Fresh Godot 4.7.2 scenes 42–43 (24 static frames) and 44–45 (96 moving frames) were captured after the training and threshold were fixed. Both datasets passed pair and geometry validation. Evaluation used CPU for both checkpoints on identical scenes; model weights came from the MPS training run.

| Fresh split | Cheap MAE | Prior recurrent MAE | Preserved gate MAE | Cheap temporal error | Prior temporal error | Preserved temporal error | Preserved false darkening |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static 42–43 | 0.009678 | 0.008633 | 0.008698 | 0.005692 | 0.006726 | 0.006264 | **0.110%** |
| Moving 44–45 | 0.014665 | 0.013200 | 0.013264 | 0.007341 | 0.008058 | 0.007908 | **0.263%** |

The preservation-trained gate reduced temporal error against the prior recurrent checkpoint, but slightly worsened frame MAE. It failed the 0.1% preservation limit on both fresh splits, and temporal error still exceeded cheap rendering. The prior checkpoint also exceeded the preservation limit on these new scenes. This is evidence of poor scene-level generalization, not a passing upgrade. Local evaluation records are `artifacts/{preserved,baseline}_{static,motion}_fresh.json`; the fresh datasets are ignored by Git.

The next investigation should measure false-darkening by scene and by distance from shadow boundaries, then expand training/validation scene diversity before tuning another loss or layer. These four fresh scenes are now used and should not become the next selection set.

## Expanded-scene iteration, 2026-11-20

The per-scene diagnostic localized the previous fresh failure: scene 45 had 0.496% false darkening, while scene 44 had 0.033%. Across those moving frames, unchanged pixels within three pixels of a changed-reference region had 0.646% false darkening, versus 0.019% farther away. This is an observed concentration near shadow boundaries, not proof of a single root cause. The current model output can be inspected in `artifacts/current_recurrent_output.png` using `python3 -m evaluation.visualize_recurrent --frame 44 0 --frame 44 16 --frame 44 32 --frame 45 24`.

Scenes 42–43 and 44–45, which had already been evaluated, were added to new static and motion training manifests. They are no longer held-out scenes. The original validation scenes 24–27 remained fixed. A three-epoch MPS run began from `shadow_refine_confidence_preserved_mps.pt` with the same loss weights, bias 5, and 0.1% validation cap. Epoch 1 was the best eligible checkpoint. The new checkpoint is `artifacts/shadow_refine_confidence_expanded_mps.pt` (local and Git-ignored).

Fresh Godot scenes 46–47 (24 static frames) and 48–49 (96 moving frames) were captured only after training and selection. Both datasets passed capture validation. The expanded checkpoint and its direct parent were evaluated on the same scenes at bias 5:

| Fresh split | Cheap MAE | Parent MAE | Expanded MAE | Cheap temporal error | Parent temporal error | Expanded temporal error | Parent false darkening | Expanded false darkening |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static 46–47 | 0.012745 | **0.010136** | 0.010387 | **0.006597** | 0.007155 | 0.007147 | 0.0545% | **0.0373%** |
| Moving 48–49 | 0.009686 | **0.008375** | 0.008505 | **0.005917** | 0.006151 | 0.006100 | 0.1357% | **0.1014%** |

More scene data reduced false darkening and slightly reduced temporal error, but frame MAE worsened and the moving split narrowly missed the 0.1% cap. Temporal error still exceeds cheap rendering. This candidate is therefore not a validated upgrade. The new scenes are now used for evaluation; subsequent tuning needs another fresh set. Local result files are `artifacts/{expanded,parent}_{static,motion}_iteration.json`.

Reproduce the local combined manifests and training with these commands. The manifest tool links existing local dataset paths; it does not copy images:

```sh
python3 -m training.combine_manifests --output dataset/expanded_static_train --source dataset/shadow_resolution train train --source dataset/shadow_resolution validation validation --source dataset/preservation_static_fresh test train
python3 -m training.combine_manifests --output dataset/expanded_motion_train --source dataset/motion_train train train --source dataset/preservation_motion_fresh test train
python3 -m training.train_recurrent --data dataset/expanded_static_train --motion-data dataset/expanded_motion_train --validation-data dataset/motion_validation --init-checkpoint artifacts/shadow_refine_confidence_preserved_mps.pt --output artifacts/shadow_refine_confidence_expanded_mps.pt --device mps --epochs 3 --batch-size 1 --temporal-weight 10 --confidence-gate --confidence-weight 0.25 --preservation-weight 5 --selection-mask-bias 5 --max-false-darken-rate 0.001
```

## Boundary weighted loss iteration, 2026-11-20

The next iteration replaced uniform preservation weighting with a 5× weight near reference shadow boundaries, using a three-pixel radius. The expanded-scene checkpoint was fine-tuned for three MPS epochs; epoch 1 was selected by the existing validation objective and preservation cap. New Godot scenes 50–51 (static) and 52–53 (moving) were captured for test after selection. Both datasets passed capture validation. At bias 5, the parent and boundary-weighted models produced:

| Fresh split | Cheap MAE | Parent MAE | Boundary MAE | Cheap temporal error | Parent temporal error | Boundary temporal error | Parent false darkening | Boundary false darkening |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static 50–51 | 0.013203 | 0.011304 | **0.011042** | **0.006912** | 0.007647 | 0.007867 | **0.0638%** | 0.1096% |
| Moving 52–53 | 0.014946 | 0.013001 | 0.012975 | **0.007201** | 0.007579 | 0.007569 | **0.0410%** | 0.0457% |

The boundary loss slightly improved frame MAE on both splits, but it raised false darkening above the 0.1% limit on static scenes and raised temporal error on static scenes. The moving split remains below the preservation limit, but temporal error is still above cheap rendering. This iteration also does not meet the full acceptance criteria.

## Stopping point

Across repeated scene-level holdouts, the model consistently lowers frame MAE relative to the cheap render on these procedural scenes. The temporal models consistently have higher temporal error than the cheap render, and preservation performance varies by scene and split. Two targeted history/preservation changes and an expanded-scene training run did not satisfy the joint targets for frame quality, temporal stability, and false darkening. We should stop model tuning here and present this as a completed proof of concept: the chosen renderer inputs support learned shadow refinement, but robust temporal preservation and real-time end-to-end performance remain unproven. The checkpoints and generated datasets remain local; evaluation JSONs are under ignored `artifacts/`.
