# AirVLA implementation checklist

**Start here.** This is our working checklist. Use the [technical reference](AirVLA_IsaacLab_Reproduction_Guide.md) for design details, data conventions, and paper settings.

**Current status:** M0–M4 complete. The all-16-episode server training and repeated M5 evaluation are complete, but learned control remains unsuccessful. Both step 7,500 and step 10,000 scored **0/2 familiar and 0/4 validation successes**, with one closed-gripper contact event each and no lifts. Fresh expert replay and both execution audits passed. See the [comparison report and videos](artifacts/M5/baseline_comparison.md); the [original debug report](artifacts/M5/report.md) is preserved.

**Diagnosis complete:** [Failure diagnosis](artifacts/M5/diagnosis.md) found hidden expert timing requirements, delayed/missed approaches, and gripper closure without reliable alignment. All 71 sampled requests reproduced their saved predictions exactly; fixed-noise RGB/state interventions show sensitivity, not proof of visual grounding.

**Corrective data ready (2026-10-19):** The separate state-driven expert passed 5/5 physical pilot cases (including failed closure and loss of grip at lift onset) and 16/16 collection episodes. The new dataset has 18 training episodes / 6,534 rows, including two fault-free recovery suffixes, plus the four unchanged validation episodes / 2,068 rows. All 13 dataset checks passed. See [report, videos and commands](artifacts/M5/recovery_report.md).

**Next task:** Run a small learning pilot using `configs/train_recovery.yaml`, then fixed M5 closed-loop evaluation before another full server run. The disposable 25-step batch-4 probe passed on server port 30 on 2026-10-21; the recovery dataset was transferred and all 17 file checksums matched. This configuration starts from `pi0_base` with `datasets/airvla_m5_recovery` and its new normalization. No full recovery training run or trained checkpoint is ready yet. Old validation targets retain timed holds, so prioritize physical task success over offline error alone. Broader yaw/layout coverage and an untouched/OOD test suite remain future work; defer RTC until useful baseline success.

```bash
bash scripts/run_lerobot.sh scripts/m4_train.py \
  --config configs/train_recovery.yaml --probe --probe-steps 25
```

**Historical baseline training command** (run from `/home/user/pivla`; the launcher selects `lerobot` automatically):

```bash
bash scripts/run_lerobot.sh scripts/m4_train.py --config configs/train.yaml
```

**Server command** (`zhiyang@47.116.73.163`, SSH port 30): activate the server environment and use the server launcher. It uses the active Python (or `LEROBOT_PYTHON`) and the project's local dataset/model files. Select a free GPU with `nvidia-smi` before starting; the example selects GPU 0.

```bash
conda activate lerobot
cd ~/workspace/code/pivla
CUDA_VISIBLE_DEVICES=0 bash scripts/run_lerobot_server.sh scripts/m4_train.py \
  --config configs/train.yaml --output artifacts/M5/baseline_train_server
```

For a short check first, add `--probe --probe-steps 10` and use a fresh output directory such as `artifacts/M5/server_probe_manual`. The server launcher sets offline Hugging Face access and resolves project imports without the local machine's absolute paths. Keep `datasets/airvla_m3`, `checkpoints/pi0_base`, and `checkpoints/paligemma_tokenizer` under the project directory.

**Live training display:** New runs show an updating terminal panel with loss, a rolling loss average, learning rate, gradient norm, speed, peak GPU memory, evaluation results, and optimizer ETA. Redirected output uses readable log lines; `--plain` also selects that format. Every step is still recorded in `loss.jsonl`.

To inspect the baseline logs, open a second SSH terminal on **port 31**, activate `lerobot`, and run from `~/workspace/code/pivla`:

```bash
python scripts/watch_training.py artifacts/M5/baseline_train_server
```

The monitor reads existing logs and uses no GPU. `Ctrl-C` closes only the monitor; `--once` prints one snapshot. The existing training process keeps its original console format until its next launch. ETA covers remaining optimizer steps, excluding future evaluation and checkpoint saves.

[Training settings](configs/train.yaml): all 8,188 training rows, batch 4, 10,000 steps, 500-step warmup to `5e-5`, then cosine decay to `5e-6`. Start from `pi0_base`; train the action expert/projections with BF16 and gradient checkpointing. Validation and checkpoints run every 2,500 steps, using the unchanged training-only normalization. The user launched the full run on the server; it completed 10,000 steps with a successful checkpoint reload.

Batch size was reduced from 8 to 4 at the user's request, keeping the 10,000-step budget. This processes approximately 40,000 samples, or 4.9 passes through the training rows. The earlier batch-8 benchmark measured 0.851 s/step, 11.88 GiB allocated and 12.51 GiB reserved CUDA memory; it is historical evidence, not a batch-4 duration estimate. Batch-4 throughput has not been measured locally.

**Server readiness (2026-10-14):** The pinned model, tokenizer, and dataset passed checksum checks; train/validation samples and a ten-step batch-4 probe passed on an RTX PRO 5000 72GB GPU. The probe measured 0.476 s/step after warmup and 10.97 GiB peak allocated memory. Its 79-minute projection for 10,000 optimizer steps excludes validation, checkpoint saves, startup, and other server activity. See [setup results](artifacts/server_setup/setup_result.json). The user subsequently completed full training on port 31, using the same shared project files. Optimizer time was 5,033 s (about 84 minutes), with 10.97 GiB peak allocated memory. Validation position-command RMSE was 5.97 cm at step 7,500 and 7.87 cm at step 10,000; these offline scores do not establish task success. See [training result](artifacts/M5/baseline_train_server/result.json).

[Readiness checks](artifacts/M5/training_setup.json) passed: batch-8 benchmark, two schedule/RNG tests, and a separate two-step run exercising validation, intermediate/final saves, and a fresh reload with identical predictions. Shared environments were unchanged.

Output: `artifacts/M5/baseline_train/`. Intermediate checkpoints: `checkpoints/step_002500`, `step_005000`, and `step_007500` inside that directory; final checkpoint: `checkpoint/`. Follow `loss.jsonl` for loss/LR and `evaluation.jsonl` for training/validation prediction metrics. Existing output directories are rejected; use `--output artifacts/M5/baseline_train_v2` for another run. Saved checkpoints contain weights and preprocessing, not optimizer state; `--checkpoint` starts a fresh optimizer and schedule.

**First useful result achieved:** The drone physically grasps, lifts, transports, and releases a 100 g block into a bin, with three recorded camera streams and flight/contact measurements.

**Environment choice and constraint:** Use the existing `isaaclab` environment for simulation and the existing `lerobot` environment for dataset/policy work, as requested. On 2026-10-10 the user authorized modifying `lerobot` to meet project needs. Ten missing π0 dependency packages were added, preserving all 111 existing package versions. Keep `isaaclab`, shared source files, editable links, and shell configuration unchanged. Use the project launchers for local imports/runtime files; no environment clone is planned.

## How we work through this

1. Follow the **Next task** above; otherwise work on the first incomplete milestone. Read its linked reference section when needed.
2. Implement the smallest working version and run its completion checks.
3. Save the command, configuration, and evidence under `artifacts/M<number>/`.
4. Mark a milestone complete only when its checks pass. Update the progress log before moving on.

The external project lives at `airvla_isaac/`. M1 is implemented in `source/airvla_isaac/airvla_isaac/sim/` inside that project, and runs through `scripts/m1_hover.py`. The generated direct cartpole task is retained as the M0 installation check. Later milestone code paths remain planned responsibilities within the package. Shared configs and evidence are under this workspace's `configs/` and `artifacts/`.

## Milestone map

| Done | Milestone | Depends on | Result |
|:--:|:--|:--|:--|
| [x] | [M0 — Setup](#m0--setup) | — | Verified versions and working simulator |
| [x] | [M1 — Hover and cameras](#m1--hover-and-cameras) | M0 | Force-driven hover and three RGB views |
| [x] | [M2 — Scripted manipulation](#m2--scripted-manipulation) | M1 | Physical grasp, lift, transport, and release |
| [x] | [M3 — Dataset](#m3--dataset) | M2 | Validated demonstrations and action conversion |
| [x] | [M4 — Training smoke test](#m4--training-smoke-test) | M3 | Reloadable π0 checkpoint and offline predictions |
| [x] | [M5 — Policy baseline](#m5--policy-baseline) | M4 | Debug checkpoint measured: 0/6 successes; interface audit passed |
| [ ] | [M6 — RTC](#m6--rtc) | M5 | Overlapping inference with consistent actions |
| [ ] | [M7 — Payload guidance](#m7--payload-guidance) | M6 | Tested guidance and a controlled comparison |
| [ ] | [M8 — Navigation and composition](#m8--navigation-and-composition) | M7 | Recovery-data and composed-task evaluations |

**For the first working demo, focus on M0–M2.** Detailed textures, MPC, explicit motor models, 3D Gaussian Splatting (3DGS), real hardware, and a world model are later extensions.

## M0 — Setup

**Build / check**

- [x] Record OS, GPU model/VRAM, driver, available disk, and existing simulator installations.
- [x] Select and record the existing Isaac Sim / Isaac Lab pair and Python, PyTorch, and LeRobot versions/commits in [configs/versions.yaml](configs/versions.yaml), without upgrading shared installations.
- [x] Confirm separate Isaac Lab and LeRobot environments and CUDA operation in each. Full π0 dependency/model verification remains an M4 prerequisite.
- [x] Run bounded copies of the empty-scene and camera examples; inspect the saved camera image.
- [x] Generate an external direct-workflow project without package installation; run its example for 120 steps.

**Done when:** The empty scene steps successfully, the camera example saves an inspectable image, and the external project launches. Save terminal output, the image, and exact commands in `artifacts/M0/`.

**Evidence:** [M0 report and commands](artifacts/M0/report.md), [camera preview](artifacts/M0/camera_preview.png), and [external-project result](artifacts/M0/project_result.json). Package metadata and shared source status were unchanged after the checks.

**Follow-up:** M1 is complete. Training uses the existing `lerobot` environment. The originally missing `transformers` dependency was resolved during authorized M4 environment preparation; see the [environment report](artifacts/M4/environment_setup.md).

Reference: [Software and startup](AirVLA_IsaacLab_Reproduction_Guide.md#2-software-and-startup).

## M1 — Hover and cameras

**Build:** `sim/scene_cfg.py`, `sim/drone_cfg.py`, `sim/flight_controller.py`, `sim/cameras.py`, `sim/env.py`, and `configs/sim.yaml`.

- [x] Create one simple room, a dynamic drone, and a deterministic reset. Use simple collision geometry.
- [x] Apply thrust and body torque; implement position/yaw tracking with a geometric PD controller.
- [x] Record world/body frames, quaternion ordering, masses, controller gains, and actuator limits.
- [x] Add front, downward, and fixed external cameras. Save a frame from each and verify what each sees.
- [x] Log pose, target pose, velocity, thrust, torque, simulation time, and camera timestamps.

**Done when:** An unloaded drone holds a fixed target for 20 simulated seconds after settling; all camera streams are usable; three same-seed resets reproduce the configured initial state. Save a video and altitude/position-error plots in `artifacts/M1/`.

**Suggested local threshold:** Maximum position error ≤0.10 m during the hover window. Record any revised threshold before measuring; this is an engineering check, not a paper result.

**Evidence:** [M1 report and reproduction commands](artifacts/M1/report.md), [results](artifacts/M1/result.json), [camera preview](artifacts/M1/camera_preview.png), [hover plots](artifacts/M1/hover_plots.png), and local [video](artifacts/M1/hover.mp4). Maximum hover error was 0.003684 m over the 20 s window after 5 s settling. Three same-seed resets and one-second trajectories matched; all cameras produced 125 frames at 5 Hz. Shared environment metadata and source status were unchanged.

**Follow-up:** M2 verified payload contact with the same flight gains. Empty-mass feedforward includes the installed fingers; the grasped payload is not added to the controller's mass model.

Reference: [Scene and physics](AirVLA_IsaacLab_Reproduction_Guide.md#3-scene-and-physics) and [Cameras and timing](AirVLA_IsaacLab_Reproduction_Guide.md#4-cameras-and-timing).

## M2 — Scripted manipulation

**Build:** `sim/gripper_controller.py`, `data/expert_policy.py`, and `data/collect_demos.py`.

- [x] Attach a physical two-finger gripper. Add a table, one rigid block, and an open bin.
- [x] Implement `approach → descend → grasp → lift → transport → release → retreat`.
- [x] Advance stages using pose, aperture, and contact conditions; add timeouts and log failed stages.
- [x] First grasp and lift; then add transport and release.
- [x] Compare empty and loaded flight using the same controller settings. Include a 0.10 kg block if within the drone's thrust capacity; document a different mass if necessary.
- [x] Save one complete raw episode with images, timestamps, actual state, desired setpoints, gripper commands, and outcome.

**Done when:** The object remains held after leaving the table, physically loads the drone, and is released into the bin. Target 10/10 successful scripted runs in the fixed scene from repeatable resets. Save videos and loaded-versus-empty thrust/altitude plots in `artifacts/M2/`.

**Evidence:** [M2 report](artifacts/M2/report.md), [10/10 results](artifacts/M2/result.json), [load/contact plots](artifacts/M2/load_contact_plots.png), [camera preview](artifacts/M2/camera_preview.png), and local [video](artifacts/M2/pick_place.mp4). Each run took 51.4 simulated seconds; the recorded episode has 257 frames per camera. Mean hover thrust rose from 10.3005 N to 11.2815 N. Loaded altitude settled 0.1561 m below target with unchanged gains and sustained two-finger contact. No attachment or pose-tracking shortcut holds the block. The full M1 regression passed, and [shared environments/sources remained unchanged](artifacts/M2/environment_preservation.json).

**Stop here for the first demo.** If the expert cannot complete the task reliably, fix control/contact behavior before collecting training data.

Reference: [Scene and physics](AirVLA_IsaacLab_Reproduction_Guide.md#3-scene-and-physics) and [Expert demonstrations](AirVLA_IsaacLab_Reproduction_Guide.md#6-expert-demonstrations).

## M3 — Dataset

**Build:** `data/convert_to_lerobot.py`, `data/validate_dataset.py`, `policy/action_codec.py`, and `configs/dataset.yaml`.

- [x] Freeze the observation/action convention from reference section 5 in configuration.
- [x] Keep absolute expert setpoints in raw recordings and dataset rows; construct relative actions using one anchor pose per sampled chunk.
- [x] Verify action conversion, yaw wrapping, gripper meaning, normalization, and episode-end masking before training.
- [x] Collect 20–30 varied episodes for debugging. Randomize initial drone/object/bin poses within reachable limits and log seeds.
- [x] Export three image streams, state, actions, and task text to LeRobotDataset; keep the timestamped raw recordings.
- [x] Split by episode/scenario before computing training statistics. Prevent any chunk from crossing an episode boundary.
- [x] Review five full episodes and compare reconstructed action targets against logged absolute setpoints. Every down/external frame was inspected in chronological contact sheets; full three-camera videos are saved for playback.

**Done when:** The loader yields correctly shaped observation/chunk batches; images never come from the future; decoded targets match the raw targets within float precision; padding and split checks pass. Save the schema, validation report, split manifest, and five preview videos in `artifacts/M3/`.

**Evidence:** [M3 report](artifacts/M3/report.md), [validation checks](artifacts/M3/validation_report.json), and [reconstruction plot](artifacts/M3/action_reconstruction.png). Twenty successful episodes yielded 8,188 training rows and 2,068 validation rows. All 13 validation checks passed; maximum action round-trip error was 5.96e-8. Existing environments and shared sources were preserved.

**Then:** Expand toward 100–200 useful episodes for a baseline as needed. These counts are development estimates, not guaranteed learning requirements.

Reference: [Data and action contract](AirVLA_IsaacLab_Reproduction_Guide.md#5-data-and-action-contract).

## M4 — Training smoke test

**Build:** The training preprocessing integration, `configs/train_debug.yaml`, and `configs/train.yaml`.

**Environment preparation passed:** Transformers 5.5.4 and its missing dependencies are installed. π0 imports, CUDA BF16 computation, dataset loading, approved tokenizer access, and strict pretrained checkpoint loading all pass. The real forward/backward/optimizer probe passed at batch size 2.

**Evidence:** Four adapter tests passed. After 600 initial steps and 1,000 refinement steps, offline prediction errors passed the predeclared limits: position 0.080446 m, yaw 0.009563 rad, and gripper accuracy 98.12%. Peak allocated CUDA memory was 10.708 GiB. The checkpoint reload produced identical predictions on two fixed-noise inputs; loss and prediction plots were inspected. See the [M4 report](artifacts/M4/report.md). The vision-language backbone was frozen for this debugging run.

- [x] Verify the separate LeRobot environment, checkpoint access, GPU memory use, and installed CLI/API.
- [x] Load `lerobot/pi0_base`; verify three-camera mapping and the 8-state / 7-action adapter.
- [x] Confirm the relative-action conversion runs before normalization and uses statistics from the training split.
- [x] Deliberately overfit a few demonstrations. Start with a small batch and short run, then adjust from measured memory and convergence.
- [x] Save/reload the checkpoint together with preprocessing, normalization statistics, feature names, and action convention.
- [x] Plot predicted versus expert unnormalized trajectories and gripper transitions on those training episodes.

**Done when:** Training runs without invalid values, the saved checkpoint reloads, and offline predictions reproduce the small training set's motion and gripper sequence within documented tolerances. Loss decreasing alone is insufficient. Save the config, loss curve, prediction plots, and checkpoint location in `artifacts/M4/`.

**Then:** Run the baseline training configuration above. Record the freeze settings consistently in later comparisons.

Reference: [Training and deployment](AirVLA_IsaacLab_Reproduction_Guide.md#7-training-and-deployment).

## M5 — Policy baseline

**Build:** `policy/pi0_adapter.py`, `policy/rollout.py`, `eval/evaluate.py`, `eval/metrics.py`, and `configs/eval.yaml`.

- [x] Replay expert action chunks through the deployment adapter first; they must reconstruct the intended flight targets.
- [x] Run naïve π0 with the same cameras, state ordering, and action decoder used in training.
- [x] Keep physics and the flight controller running while inference executes. Define the queue-empty behavior explicitly.
- [x] Record action horizon, executed actions per inference, wall-clock inference delay, simulation-time mode, and any fallback use.
- [x] Freeze evaluation seeds, payload masses, success thresholds, and timeouts before comparing methods.
- [x] Run held-out trials and save success counts, failures, trajectories, and videos.

**Evidence:** [M5 report](artifacts/M5/report.md), [results](artifacts/M5/result.json), and [execution audit](artifacts/M5/verification.json). Expert replay placed the block successfully; the learned policy achieved no grasps in six trials. All 3,118 executed commands matched the decoded model outputs, and physics continued during inference. The 50-action horizon executes ten commands per request, holding the last target while waiting; median inference was 92.83 ms. This is a measured debug baseline, not full-dataset training or a useful learned manipulation policy. Improve it before RTC comparisons.

**Done when:** There is a reproducible baseline report and rollout video, with conversion/timing faults ruled out. Report low task success honestly; do not use RTC to hide a broken interface. Save results in `artifacts/M5/`.

Reference: [Training and deployment](AirVLA_IsaacLab_Reproduction_Guide.md#7-training-and-deployment) and [Evaluation](AirVLA_IsaacLab_Reproduction_Guide.md#9-evaluation).

## M6 — RTC

**Build:** `policy/rtc.py` as an integration layer around supported sampler/queue APIs where possible.

- [ ] Check RTC support in the pinned LeRobot revision before implementing it yourself.
- [ ] Generate the next chunk asynchronously while executing the current one.
- [ ] Preserve actions committed during inference and condition the new suffix inside the sampler.
- [ ] Re-express overlapping actions relative to the new observation's anchor before sampler conditioning.
- [ ] Test delayed inference, queue exhaustion, reset, and episode changes. Reject stale results.
- [ ] Compare against M5 using the same checkpoint and evaluation trials; record any horizon changes.

**Done when:** Logged setpoints show correct overlap/anchor handling; committed actions are preserved; resets clear all queued state; the baseline/RTC comparison is saved in `artifacts/M6/`. Record boundary discontinuities and latency, even if RTC does not improve task success.

Reference: [RTC and payload guidance](AirVLA_IsaacLab_Reproduction_Guide.md#8-rtc-and-payload-guidance).

## M7 — Payload guidance

**Build:** `policy/payload_guidance.py` and its integration with the π0 flow sampler.

- [ ] Implement payload confidence from gripper command history and measured aperture.
- [ ] Compute vertical guidance through the sampler, including action denormalization and the chunk anchor.
- [ ] Check gradient finiteness and that the correction moves the physical altitude target in the intended direction.
- [ ] With identical observations/noise, verify zero guidance reproduces the unguided sampler within numerical tolerance.
- [ ] Compare RTC versus RTC + guidance across the same seeds and payload masses; keep the checkpoint and controller fixed.

**Done when:** Sampler checks pass and the comparison reports altitude sag, grasp retention, control effort, and stage/task success. Save plots and results in `artifacts/M7/`. A measured lack of benefit is a valid result.

Reference: [RTC and payload guidance](AirVLA_IsaacLab_Reproduction_Guide.md#8-rtc-and-payload-guidance).

## M8 — Navigation and composition

**Build:** Gate-task expert logic, corrective-rollout generation, and navigation/composition evaluation configurations.

- [ ] Establish reliable scripted gate passage and hover first.
- [ ] Collect navigation demonstrations and extra recovery demonstrations from perturbed approaches/waypoints.
- [ ] Train and compare baseline navigation data against the same data plus corrective rollouts, with matched training budgets.
- [ ] Train a mixed atomic-task policy on navigation and manipulation demonstrations.
- [ ] Evaluate a composed instruction absent from training: navigate through the gate, then pick and place.
- [ ] Report gate, hover, pick, place, and full-task results on held-out scenarios.

**Done when:** The experiment matrix in reference section 9 is complete, with dataset/checkpoint IDs, seed lists, configurations, counts, failure causes, and videos in `artifacts/M8/`.

Native Isaac rendering is the first implementation. A later 3DGS experiment needs its own renderer comparison; simulator recovery data alone does not reproduce the paper's 3DGS contribution.

Reference: [Demonstrations](AirVLA_IsaacLab_Reproduction_Guide.md#6-expert-demonstrations) and [Evaluation](AirVLA_IsaacLab_Reproduction_Guide.md#9-evaluation).

## Progress log

Update this after each work session. Use `not started`, `in progress`, `blocked`, or `done`; link evidence instead of relying on memory.

| Date | Milestone | Status | Evidence / blocker | Next action |
|:--|:--|:--|:--|:--|
| 2026-10-09 | M0 | Done | [Report](artifacts/M0/report.md): 120 empty-scene steps, 20 RGB frames, 120 direct-task steps, CUDA in both environments; shared package metadata/source status unchanged | M1: drone physics, hover controller, and three cameras |
| 2026-10-10 | M1 | Done | [Report](artifacts/M1/report.md): 20 s hover, maximum error 3.684 mm, three repeatable resets, 125 frames per camera, free-fall and controller checks passed; shared environment/source fingerprints unchanged | M2: physical gripper and scripted grasp-and-lift |
| 2026-10-10 | M1 GUI fix | Done | [GUI validation](artifacts/M1/gui_validation.json): corrected render-driven extra physics steps; full desktop run passed with exactly 5,000 engine ticks and 3.684 mm maximum hover error | M2; use `--gui --keep-open` to inspect the simulator |
| 2026-10-10 | M2 | Done | [Report](artifacts/M2/report.md): physical 100 g grasp/transport/release, 10/10 fixed-scene runs, 0.981 N extra hover thrust, three-camera video and raw episodes | M3: action conversion and varied LeRobot demonstrations |
| 2026-10-10 | M3 | Done | [Report and commands](artifacts/M3/report.md): 20/20 successful varied episodes, 16/4 scenario split, 10,256 rows, all 13 dataset checks passed, five reviewed previews; shared environments unchanged | M4: verify π0 readiness, integrate preprocessing, overfit a few demonstrations |
| 2026-10-10 | M4 environment preparation | Done | [Environment report](artifacts/M4/environment_setup.md): authorized installation of ten missing packages; 111 existing versions preserved; π0 imports, CLI, CUDA BF16, dataset loading and pip checks passed | Continue M4 with model/tokenizer access, preprocessing integration, and training |
| 2026-10-10 | M4 | Done | [Report](artifacts/M4/report.md): two-episode overfit passed after 1,600 steps; 8.04 cm command RMSE, 98.12% gripper accuracy, 10.708 GiB peak memory, exact reload predictions | Establish baseline training settings, then M5 expert replay and learned-policy closed-loop evaluation |
| 2026-10-10 | M5 debug baseline evaluation | Done; policy unsuccessful | [Report and videos](artifacts/M5/report.md): expert replay passed; 0/2 familiar and 0/4 held-out learned successes; eight adapter/deployment tests and execution audit passed | Improve training/data coverage and repeat M5 before M6; full-finetuning remains unrun |
| 2026-10-10 | All-16-episode training setup | Ready; full run not started | [Readiness](artifacts/M5/training_setup.json): batch 8 fits; 0.851 s/step, 11.88 GiB allocated; validation/save/reload smoke test passed | User starts the training command above, then evaluate the new checkpoint with M5 |
| 2026-10-10 | Local batch preference | Configured | Reduced batch size to 4 at user request; retained 10,000 steps and all other training settings. Earlier benchmark used batch 8. | Use the same training command; batch-4 timing remains unmeasured |

| 2026-10-14 | All-16-episode server training | Done | 10,000 steps, batch 4, 84 minutes optimizer time, exact checkpoint reload; [result](artifacts/M5/baseline_train_server/result.json) | Both evaluated; see the comparison below |

| 2026-10-14 | M5 all-16-episode comparison | Done; policies unsuccessful | [Report and videos](artifacts/M5/baseline_comparison.md): both checkpoints 0/6 successes, no lifts; fresh replay, eight tests, both execution audits and video checks passed | Diagnose approach/grasp failures; prepare corrective data before further training or RTC |

| 2026-10-14 | M5 failure diagnosis | Done | [Diagnosis](artifacts/M5/diagnosis.md): 71 exact prediction replays and fixed-noise input tests; near-identical hover observations have different upcoming targets due to expert waits | Implement and validate state-driven expert and training-only recovery collection |

| 2026-10-19 | State-driven expert and corrective data | Done | [Report](artifacts/M5/recovery_report.md): 5/5 physical pilot cases, 16/16 collection episodes; 18 training episodes with recovery suffixes, 4 unchanged validation episodes; all 13 dataset checks passed | Smoke test and small learning pilot with `configs/train_recovery.yaml`; new data not yet trained |
| 2026-10-21 | Recovery server readiness | Done | [Evidence](artifacts/M5/recovery_result.json): dataset transfer verified; 25-step batch-4 probe passed on port 30 GPU 0, 0.475 s/step and 10.97 GiB peak allocated memory; use `run_lerobot_server.sh` | Small learning pilot and fixed M5 evaluation; full recovery training not started |

To start a future session:

> Read `implement_guide.md` and its linked technical reference. Follow its **Next task**, then the first incomplete milestone. Implement it, run its completion checks, and update the progress log with commands, evidence, and blockers. Keep proposed work distinct from verified results.
