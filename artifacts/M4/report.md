# M4 — π0 training smoke test

Status: **passed on 2026-10-10**. Fine-tuned the π0 action expert/projections on two demonstrations for 1,600 total steps, passed the predeclared offline prediction criteria, saved the model and preprocessing contract, and reloaded it with identical predictions. This is a two-episode debugging checkpoint, not a validated general policy or a closed-loop flight result.

## Results

| Metric | Pretrained | After 600 steps | Final, 1,600 steps | Required |
|---|---:|---:|---:|---:|
| Position-vector RMSE | 0.314223 m | 0.110813 m | **0.080446 m** | ≤0.10 m |
| Yaw RMSE | 0.015207 rad | 0.012044 rad | **0.009563 rad** | ≤0.10 rad |
| Gripper accuracy | 40.20% | 96.80% | **98.12%** | ≥95% |

Evaluation uses 62 observation/chunk starts and 2,878 valid future actions from the two training episodes, including both sides of gripper transitions and episode ends. The 1,024 training rows come exclusively from train episodes 0 and 4. Position error is the RMSE of the three-dimensional desired-position vector, not a measured flight tracking error. Gripper accuracy thresholds the unnormalized output at 0.5. No held-out episode was used to choose statistics or train the model.

All 1,600 losses and gradient norms were finite, with occasional loss spikes. The two training phases took 149.5 + 248.8 = 398.3 seconds excluding model loading, evaluation, and saving; peak allocated CUDA memory was 10.708 GiB. Both checkpoint reloads reproduced predictions exactly on the two fixed-noise reload-check inputs (maximum error 0.0).

Evidence: [final results](result.json), [complete loss curve](loss.png), [offline position/yaw/gripper predictions](predictions.png), and [environment preservation](training_environment_preservation.json). Both episode plots were visually inspected: predictions follow descent, lift, transport, release, and retreat setpoints with residual error. The trajectory plot displays the first predicted action at each evaluated observation; the metrics assess every valid future action in each sampled chunk.

**Final checkpoint:** `artifacts/M4/debug_refine/checkpoint/`. It contains `model.safetensors`, model config, tokenizer, M3 schema, training normalization statistics, and the adapter/camera mapping. Use the project adapter and strict loader with this checkpoint; the generic π0 processor does not implement this drone's custom state/action convention.

The first run remains marked failed because its 0.110813 m position error exceeded the 0.10 m limit. Its checkpoint and evidence remain under `debug_run/`; the passing refinement is in `debug_refine/`. Tolerances and evaluation rows were unchanged. Refinement started from saved weights with a fresh optimizer, rather than resuming optimizer state.

## Completed work

- [Environment preparation](environment_setup.md): π0 dependencies installed with authorization; all 111 prior package versions preserved and Isaac Lab untouched.
- Downloaded [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base), pinned to revision `6e245399ad403ccf02b8112b73cfdc2633068f38`. Weights occupy 14,005,618,584 bytes in `checkpoints/pi0_base/`, excluded from Git. [Provenance](model_provenance.json) records the weight checksum.
- [Strict model loading passed](model_load.json). Model construction/loading peaked at 8.287 GiB allocated CUDA memory. The constructed model has 4,028,019,472 parameters, of which 578,036,768 are trainable with the debug freeze configuration. All three actual images preprocess to `[1,3,224,224]`; state and actions pad to `[1,32]` and `[1,50,32]`. This is model-loading memory only, not a forward/backward or optimizer-step measurement.
- Project-local adapter implemented in `policy/pi0_adapter.py`. Camera mapping is front → base, down → left-wrist slot, external → right-wrist slot. These are model input slots, not claims that the drone has wrists. All three real images are used; π0 performs its own 224×224 resizing and image scaling.
- The adapter retains M3's 8-state/7-action contract. `ChunkDataset` alone performs relative conversion and numeric normalization; generic π0 relative conversion and normalization are disabled. Predictions are decoded using the saved observation anchor and training statistics.
- The current upstream π0 `forward()` averages its loss without applying `action_is_pad`. The local training loss excludes padded time steps and the model's extra padded dimensions, without changing shared LeRobot code.
- The current upstream π0 loader catches weight-loading exceptions and can return a model without loaded weights. Our local loader requires a complete weights file and a strict state-dictionary match.
- Four adapter tests passed: actual M3 camera/numeric mapping and decoding, zero loss/gradient for padded targets, configuration guards against duplicate preprocessing, and preservation of an existing trained embedding during checkpoint remapping. The tokenizer used in the adapter unit test is an explicit stub; the separate real-model probe uses the downloaded tokenizer.
- The real one-step probe passed at batch size 2: loss 0.169725, finite gradient norm 5.213834, and peak allocated CUDA memory 10.706 GiB. Evidence is retained in `debug_run_probe/result.json`.

## Tokenizer access resolved

The configured tokenizer is [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), pinned to revision `35e4f46485b4d07967e7e9935bc3786aad50687c`. Access initially failed because the CLI was unauthenticated. The user completed browser login with an approved account, and all three required tokenizer JSON files downloaded successfully to `checkpoints/paligemma_tokenizer/`. No tokens are recorded in this project.

The existing shell's `ALL_PROXY`/`all_proxy` use a `socks://` scheme rejected by this Hugging Face HTTP client. HTTP/HTTPS proxy variables already use supported HTTP URLs. For login, clear only the unsupported variables in the child process:

```bash
env -u ALL_PROXY -u all_proxy /home/user/miniconda3/envs/lerobot/bin/hf auth login
```

No shell profile or global proxy configuration was changed. Downloads run outside the offline runtime launcher. The successful download command was:

```bash
env -u ALL_PROXY -u all_proxy /home/user/miniconda3/envs/lerobot/bin/hf download \
  google/paligemma-3b-pt-224 tokenizer.json tokenizer_config.json special_tokens_map.json \
  --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
  --local-dir checkpoints/paligemma_tokenizer
```

## Experiment and reproduction

`configs/train_debug.yaml` selects training episodes 0 and 4, batch size 2, 600 steps, BF16, gradient checkpointing, and training of the action expert/projections with the vision-language backbone frozen. This is a debugging configuration. `configs/train.yaml` keeps a separate proposed full-finetuning baseline; its memory requirements and learning behavior are unverified.

`scripts/m4_train.py` first loads the real local tokenizer and pretrained weights. `--probe` performs one forward/backward/optimizer step and records peak allocated CUDA memory. The full run compares fixed-noise predictions before/after training, saves the model plus tokenizer/schema/normalization/adapter contract, and reloads it for a prediction comparison. It includes gripper-transition boundaries and episode ends in evaluation. Fresh output directories prevent accidental overwrites.

Completion tolerances were set before training: position-vector RMSE ≤0.10 m, yaw RMSE ≤0.10 rad, gripper accuracy ≥95%, and checkpoint-reload maximum absolute prediction difference ≤1e-6. All passed after refinement. They measure imitation on the selected training episodes, not held-out task success.

Commands used after the model/tokenizer downloads completed (choose fresh output directories if repeating on this workspace):

```bash
bash scripts/run_lerobot.sh scripts/test_pi0_adapter.py
bash scripts/run_lerobot.sh scripts/m4_train.py --probe
# Proceed only after reviewing probe memory and numerical results.
bash scripts/run_lerobot.sh scripts/m4_train.py
bash scripts/run_lerobot.sh scripts/m4_plot.py
```

The plot command reuses the existing Isaac Lab Python plotting packages inside the same sandbox if Matplotlib is absent from `lerobot`; it does not launch Isaac Sim.

The refinement and combined-plot commands were:

```bash
bash scripts/run_lerobot.sh scripts/m4_train.py \
  --checkpoint artifacts/M4/debug_run/checkpoint \
  --steps 1000 --learning-rate 0.00005 --output artifacts/M4/debug_refine
bash scripts/run_lerobot.sh scripts/m4_plot.py --run artifacts/M4/debug_refine \
  --previous-run artifacts/M4/debug_run
```

## Scope and next work

The two-episode overfit verifies the training interface and checkpoint path. It does not measure generalization, grasp success under learned control, inference delay, or continuous closed-loop stability. The full-finetuning configuration remains unrun. Establish the baseline training/data settings and perform M5's expert-chunk replay and learned-policy simulator evaluation before drawing conclusions about flight success or comparing RTC.

No additional packages were installed during this training work. `lerobot` package versions remain as recorded after the authorized dependency installation, Isaac Lab's metadata is unchanged, and all three shared source checkouts remain clean. Downloaded weights, run checkpoints, raw predictions, caches, and logs are Git-ignored; only compact evidence and plots are retained for commit.
