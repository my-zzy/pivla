# AirVLA implementation checklist

**Start here.** This is our working checklist. Use the [technical reference](AirVLA_IsaacLab_Reproduction_Guide.md) for design details, data conventions, and paper settings.

**Current status:** M0–M2 complete — setup, hover/cameras, and physical scripted pick-and-place verified. M2 passed 10/10 fixed-scene trials; see the [report](artifacts/M2/report.md) and local [video](artifacts/M2/pick_place.mp4). LeRobot dataset conversion and training have not started.

**Next task:** M3 — freeze the observation/action contract, validate relative-action conversion, and collect/export 20–30 varied demonstrations to LeRobot.

**First useful result achieved:** The drone physically grasps, lifts, transports, and releases a 100 g block into a bin, with three recorded camera streams and flight/contact measurements.

**Environment choice and constraint:** Use the existing `isaaclab` environment for simulation and the existing `lerobot` environment for dataset/policy work, as requested. Both serve other projects; preserve their packages, editable links, source files, and shell configuration during simulator work. Use `scripts/run_isaac.sh` for project-local simulation imports and runtime files. Record and resolve any missing π0 dependencies before M4; no environment clone is planned.

## How we work through this

1. Work on the first incomplete milestone. Read its linked reference section when needed.
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
| [ ] | [M3 — Dataset](#m3--dataset) | M2 | Validated demonstrations and action conversion |
| [ ] | [M4 — Training smoke test](#m4--training-smoke-test) | M3 | Reloadable π0 checkpoint and offline predictions |
| [ ] | [M5 — Policy baseline](#m5--policy-baseline) | M4 | Measured closed-loop π0 performance |
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

**Follow-up:** M1 is now complete. Training will use the existing `lerobot` environment. Its missing `transformers` dependency remains an M4 readiness item; no packages were changed by this environment selection.

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

- [ ] Freeze the observation/action convention from reference section 5 in configuration.
- [ ] Keep absolute expert setpoints in raw recordings and dataset rows; construct relative actions using one anchor pose per sampled chunk.
- [ ] Verify action conversion, yaw wrapping, gripper meaning, normalization, and episode-end masking before training.
- [ ] Collect 20–30 varied episodes for debugging. Randomize initial drone/object/bin poses within reachable limits and log seeds.
- [ ] Export three image streams, state, actions, and task text to LeRobotDataset; keep the timestamped raw recordings.
- [ ] Split by episode/scenario before computing training statistics. Prevent any chunk from crossing an episode boundary.
- [ ] Watch five full episodes and compare reconstructed action targets against logged absolute setpoints.

**Done when:** The loader yields correctly shaped observation/chunk batches; images never come from the future; decoded targets match the raw targets within float precision; padding and split checks pass. Save the schema, validation report, split manifest, and five preview videos in `artifacts/M3/`.

**Then:** Expand toward 100–200 useful episodes for a baseline as needed. These counts are development estimates, not guaranteed learning requirements.

Reference: [Data and action contract](AirVLA_IsaacLab_Reproduction_Guide.md#5-data-and-action-contract).

## M4 — Training smoke test

**Build:** The training preprocessing integration, `configs/train_debug.yaml`, and `configs/train.yaml`.

- [ ] Verify the separate LeRobot environment, checkpoint access, GPU memory use, and installed CLI/API.
- [ ] Load `lerobot/pi0_base`; verify three-camera mapping and the 8-state / 7-action adapter.
- [ ] Confirm the relative-action conversion runs before normalization and uses statistics from the training split.
- [ ] Deliberately overfit a few demonstrations. Start with a small batch and short run, then adjust from measured memory and convergence.
- [ ] Save/reload the checkpoint together with preprocessing, normalization statistics, feature names, and action convention.
- [ ] Plot predicted versus expert unnormalized trajectories and gripper transitions on those training episodes.

**Done when:** Training runs without invalid values, the saved checkpoint reloads, and offline predictions reproduce the small training set's motion and gripper sequence within documented tolerances. Loss decreasing alone is insufficient. Save the config, loss curve, prediction plots, and checkpoint location in `artifacts/M4/`.

**Then:** Run the baseline training configuration. Keep debug freezes separate from comparison settings.

Reference: [Training and deployment](AirVLA_IsaacLab_Reproduction_Guide.md#7-training-and-deployment).

## M5 — Policy baseline

**Build:** `policy/pi0_adapter.py`, `policy/rollout.py`, `eval/evaluate.py`, `eval/metrics.py`, and `configs/eval.yaml`.

- [ ] Replay expert action chunks through the deployment adapter first; they must reconstruct the intended flight targets.
- [ ] Run naïve π0 with the same cameras, state ordering, and action decoder used in training.
- [ ] Keep physics and the flight controller running while inference executes. Define the queue-empty behavior explicitly.
- [ ] Record action horizon, executed actions per inference, wall-clock inference delay, simulation-time mode, and any fallback use.
- [ ] Freeze evaluation seeds, payload masses, success thresholds, and timeouts before comparing methods.
- [ ] Run held-out trials and save success counts, failures, trajectories, and videos.

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

To start a future session:

> Read `implement_guide.md` and its linked technical reference. Work on the first incomplete milestone. Implement it, run its completion checks, and update the progress log with commands, evidence, and blockers. Keep proposed work distinct from verified results.
