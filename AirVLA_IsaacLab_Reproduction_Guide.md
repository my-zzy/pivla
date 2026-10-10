# AirVLA simulation reproduction: technical reference

**Working checklist:** [implement_guide.md](implement_guide.md) — start there for the next task, completion checks, and progress log.  
**Scope:** An AirVLA-inspired implementation in Isaac Lab, using scripted demonstrations and π0 fine-tuning.  
**Updated:** 2026-10-10. M0 setup and M1 force-driven hover/three-camera checks are verified. Physical manipulation, datasets and policy work are next. See the [M0 report](artifacts/M0/report.md) and [M1 report](artifacts/M1/report.md).

## Read only what you need

| Working on… | Read |
|:--|:--|
| Setup | [2. Software and startup](#2-software-and-startup) |
| Drone, gripper, cameras | [3. Scene and physics](#3-scene-and-physics), [4. Cameras and timing](#4-cameras-and-timing) |
| Data | [5. Data and action contract](#5-data-and-action-contract), [6. Expert demonstrations](#6-expert-demonstrations) |
| π0 | [7. Training and deployment](#7-training-and-deployment) |
| AirVLA methods | [8. RTC and payload guidance](#8-rtc-and-payload-guidance) |
| Experiments or failures | [9. Evaluation](#9-evaluation), [10. Troubleshooting](#10-troubleshooting) |

**Terms:** A VLA is a vision-language-action model. π0 is the policy we fine-tune. An action chunk is a sequence of future commands. A setpoint is a target for the flight controller. RTC means real-time chunking: generating a new chunk while the previous one executes.

## 1. What we are building

A simulated quadrotor with a gripper receives three RGB views, its state, and a language instruction. A fine-tuned π0 policy produces position/yaw targets and gripper commands; a low-level controller turns those targets into thrust and torque.

The tasks are pick-and-place, gate navigation followed by hovering, and a composed gate-then-pick-and-place instruction absent from training.

~~~text
BUILD AND TRAIN
Isaac Lab → scripted expert → raw episodes → LeRobotDataset → fine-tuned π0

RUN
RGB + state + instruction
          ↓
π0 action chunk [later: RTC + payload guidance inside sampling]
          ↓
Action decoder → position/yaw targets + gripper command
          ↓
Flight/gripper controllers → simulated physics → new observations
~~~

This project replaces the paper's real teleoperation and 3DGS rendering with scripted experts and native Isaac rendering. Results measure a simulation adaptation. Exact physical reproduction, 3DGS validation, and sim-to-real transfer remain separate experiments. See the [AirVLA project](https://airvla.github.io/).

**First milestone:** A scripted physical grasp-and-lift with three recorded camera streams. Finish the simulator and expert before adding learned control.

## 2. Software and startup

**M0–M4 have passed; M5 evaluation is complete for the debug checkpoint, with 0/6 learned task successes.** Expert replay and the deployment audit passed; see the [M5 report and videos](artifacts/M5/report.md). Improve baseline training/data coverage before RTC comparisons. M4 fine-tuned the π0 action expert on two training demonstrations, reached 8.04 cm position-command RMSE and 98.12% gripper accuracy, and verified checkpoint reload; see the [M4 report and plots](artifacts/M4/report.md). These are offline overfit results; M5 demonstrates that they do not establish successful learned flight. No deliberate OOD evaluation has been run. M3 collected 20 successful varied demonstrations, exported 10,256 rows with a 16/4 train/validation episode split, and passed all timing, action, normalization, and dataset-loader checks; see the [M3 report and previews](artifacts/M3/report.md). M2 verified physical pick-and-place with a 100 g payload in 10/10 fixed-scene runs; see the [M2 report and video](artifacts/M2/report.md). M0 verified the existing Isaac Sim 5.1.0, Isaac Lab 2.3.2 checkout, RTX 5090, and separate simulator/LeRobot environments. Exact paths and commits are in [configs/versions.yaml](configs/versions.yaml). Ten missing dependencies were added to `lerobot` with user authorization while preserving existing package versions; Isaac Lab and shared source checkouts remain unchanged. Use [the simulation launcher](scripts/run_isaac.sh) and [the offline dataset/policy launcher](scripts/run_lerobot.sh), which set imports for child processes and mount shared installations read-only.

### Proposed stack

| Component | Starting choice | Confirm before use |
|:--|:--|:--|
| OS | Ubuntu; original plan assumed 24.04 | Driver and simulator requirements on the actual machine |
| Simulator | Existing Isaac Sim 5.1.0 + Isaac Lab 2.3.2 | Verified in M0; keep recorded commits |
| Simulator Python | Python 3.11 for Isaac Sim 5.x | Simulator environment |
| Flight control | Custom geometric PD | Tracking, actuator limits, payload response |
| Training | LeRobot + PyTorch, separate environment | Commit, dependencies, GPU memory |
| Base policy | `lerobot/pi0_base` | Checkpoint access and revision |
| RTC | Supported LeRobot implementation first | API and action-convention compatibility |
| Payload guidance | Sampler integration | Gradient path and flow-time convention |

The original [Isaac Lab installation reference](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html) explains the simulator/Python setup. The local M0 record takes precedence for this machine.

### Startup commands

Run **after installation**, from the Isaac Lab repository; adjust the path. These are setup checks, not the project implementation.

~~~bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
~~~

After the empty scene works, close it and run the [camera example](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/how-to/save_camera_output.html):

~~~bash
./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py \
  --headless --enable_cameras --save
~~~

The [external project](airvla_isaac/README.md) is already generated with the direct workflow. On this shared machine, do not rerun `isaaclab.sh --new`: its wrapper installs dependencies into the active environment. M0 used the existing generator directly via `scripts/prepare_m0.py`, with no pip install. Keep the generated package structure and add these responsibilities as needed:

| Area | Planned responsibilities |
|:--|:--|
| `assets/` | Room, drone, gripper, objects |
| `sim/` | Implemented in M1: procedural scene/drone, fixed camera extrinsics, geometric PD, stepping/reset environment |
| `data/` | Expert, collection, conversion, validation |
| `policy/` | Action codec, π0 adapter, rollout, RTC, guidance |
| `eval/` | Evaluation driver and metrics |
| `configs/` | Versions, simulator, dataset, training, evaluation |
| `artifacts/` | Commands, logs, plots, videos, reports; link large datasets/checkpoints |

Original rough estimates: 3–5 days for scene/cameras, 1–2 weeks for control/manipulation, 3–7 days for data, and 1–2 weeks each for learning and AirVLA methods. Physics/contact debugging can dominate. Advance on completion checks, not elapsed time.

## 3. Scene and physics

### Minimal scene

Start with a 6 × 6 × 3 m room, floor, simple walls, table, block, bin, dynamic drone, and two-finger gripper. Add the gate for navigation. Use simple collision shapes; improve appearance after interactions work.

~~~text
Top view, not to scale
+-------------------------------------------------+
| external camera                           bin   |
|                                                 |
| drone →             gate                 table  |
|                                          block  |
+-------------------------------------------------+
~~~

### Flight and payload

Use physical thrust and torque during rollouts. Root-pose writes are for initialization/reset; using them to fly would bypass the load dynamics we need to measure.

For the proposed right-handed, Z-up world convention:

$$
m\dot v=m g_W+R_{WB}[0,0,T]^\top,\qquad g_W=[0,0,-9.81]^\top,
$$

$$
J\dot\omega+\omega\times J\omega=\tau_B.
$$

Thrust acts along body +Z; $R_{WB}$ maps body vectors to world coordinates. Adapt imported asset axes once if necessary.

Map position/yaw targets to desired acceleration/attitude, then thrust/body torque. Verify the installed wrench API's frame, application point, persistence, and stepping order. Start with geometric PD; explicit motor allocation and MPC can follow later.

Give fingers and objects collision geometry, mass, inertia, friction, and finite actuator effort. Contacts or a physically consistent constraint must transmit object forces to the drone. Visual attachment alone is insufficient.

A loaded hover should need more thrust than a comparable empty hover. Use the planned 100 g block within the drone's thrust capacity. Save aperture/contact and thrust/altitude traces with fixed controller gains. Strong disturbance rejection may leave little benefit for guidance; report it instead of degrading the controller.

## 4. Cameras and timing

| Stream | Attachment | Verify in saved images |
|:--|:--|:--|
| Front | Drone, looking forward | Corridor/gate visible |
| Down | Drone underside, looking down | Gripper/grasp region visible |
| External | Fixed room prim | Drone/work area visible |

Store each selected environment's frame as 256 × 256 × 3 uint8 RGB. Verify optical convention and parent transforms using scene landmarks; an identity quaternion does not automatically mean “forward.” Record camera calibration and transforms.

| Subsystem | Proposed rate | Interval |
|:--|--:|--:|
| Physics | 200 Hz | 0.005 s |
| Flight controller | 100 Hz | 0.010 s |
| Action setpoints | 10 Hz | 0.100 s |
| Cameras | 5 Hz | 0.200 s |
| Dataset rows | 10 Hz | 0.100 s |

The resolution and camera/action rates follow the [AirVLA system description](https://airvla.github.io/). Physics/controller rates are project choices.

At record time $t$, use each camera's newest frame **already available** at $t$. Repeated images are expected at these rates. Retain capture timestamps, availability timestamps where relevant, and image age; do not relabel reused frames as fresh. Drop startup samples until all streams are valid.

Separate simulation timestamps from wall-clock inference timing. State whether simulation continues or pauses during inference; pausing changes the latency experiment.

## 5. Data and action contract

**Proposed project convention:** Freeze this before bulk collection. The exact padding, state ordering, and chunk anchor below are implementation choices, not a fully specified paper interface.

### Features and units

| Feature | Shape | Meaning |
|:--|:--|:--|
| `observation.images.front` | 256 × 256 × 3 | Front RGB |
| `observation.images.down` | 256 × 256 × 3 | Downward RGB |
| `observation.images.external` | 256 × 256 × 3 | Fixed-room RGB |
| `observation.state` | 8 | `[x, y, z, yaw, vx, vy, vz, aperture]` |
| `action` on disk | 7 | `[x_des, y_des, z_des, 0, 0, yaw_des, g]` |
| `task` | Text | Instruction via the pinned dataset API's task mechanism |

Positions/velocities use the world frame, in metres and metres/second. Yaw is in radians about +Z. Aperture is finger separation in metres. Proposed gripper command: $g=0$ open, $g=1$ closed; it is absolute, not a delta.

The pose reference is the drone body origin. The expert handles the body-to-grasp-point offset. Record actual pose separately from desired setpoints. Document raw quaternion ordering and convert to yaw at the adapter boundary.

### One anchor for the whole chunk

Keep **absolute desired setpoints** on disk. At sample time $t$, anchor every future target to the observed body pose $(p_t,\psi_t)$:

$$
a_{t,k}=
[p^{des}_{t+k}-p_t,\;0,\;0,\;
\operatorname{wrap}(\psi^{des}_{t+k}-\psi_t),\;g_{t+k}],
\quad k=0,\ldots,H-1.
$$

Here `wrap` maps angles to $[-\pi,\pi)$. The target at $t$ is the command issued after capturing observation $t$.

Retain the same anchor during execution:

$$
p^{set}_{t+k}=p_t+\Delta p_{t,k},\qquad
\psi^{set}_{t+k}=\operatorname{wrap}(\psi_t+\Delta\psi_{t,k}).
$$

**Example:** Anchor $x_t=1.0$ m; future targets 1.2 m and 1.5 m → offsets `[0.2, 0.5]`. Add 1.0 m to both. Do not accumulate offsets or add them to the drone's changing current position.

Stacking per-row deltas gives each row a different anchor and does **not** implement this convention.

### Conversion and validation

1. Read absolute future targets and the current observation.
2. Convert position/yaw to chunk-relative targets; keep gripper absolute and padding zero.
3. Normalize using statistics computed in this representation from the training split only.
4. At inference, unnormalize once, discard padding, and decode with the stored anchor.

State yaw is index 3; action yaw is index 5. Generic elementwise subtraction of state from action is wrong.

Current LeRobot documents [relative-action preprocessing](https://huggingface.co/docs/lerobot/pi0#relative-actions). Reuse it only after verifying mapping, yaw wrapping, gripper exclusion, and statistics for this schema; otherwise write a custom processor. Never apply two relative-action conversions.

Raw recordings retain seed, task, timestamps, images, actual state, desired setpoints, gripper command/aperture, termination reason, and outcome. Log privileged object pose, contacts, mass, and collisions separately for evaluation and future world-model work; only declared observation features reach the VLA.

Use the pinned [LeRobotDataset API](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py) to create features, add frames, save episodes, and finalize. Rows contain absolute actions; the training loader applies the chunk conversion.

Split by episode/scenario before computing statistics. Keep chunks inside one episode and mask unavailable future steps. Validate image timing, feature order, gripper meaning, and absolute → relative → absolute reconstruction, including yaw near ±π.

## 6. Expert demonstrations

Use `approach → descend → grasp → lift → transport → release → retreat`. Transition on measured position error, contact, and aperture. Add stage timeouts and failure logs. Privileged state may guide the expert without becoming a policy input.

Use one instruction initially:

> Pick up the block from the table and place it in the bin.

Start with 20–30 episodes to debug recording, then expand toward 100–200 useful episodes as needed. These are development estimates. Randomize initial drone pose, object/bin placement, approach angle, and supported mass; keep evaluation scenarios separate.

For navigation, establish gate passage and hover, then perturb initial pose, approach/exit waypoints, hover height, and near-frame recovery. Generate motion through the physical controller and render its actual trajectory.

Compare the same navigation data with and without added corrective rollouts. Native rendering tests recovery-data value. A later 3DGS study needs reconstructed imagery and its own comparison against native renders.

## 7. Training and deployment

### Debug before scaling

Use the existing environment named `lerobot`, separate from the `isaaclab` simulation environment. This is the user's selected training environment; no clone is planned. The [π0 documentation](https://huggingface.co/docs/lerobot/pi0) describes dependencies and training entry points. Check `lerobot-train --help` at the pinned revision and resolve missing dependencies before M4 while respecting the shared-environment constraint.

First overfit a few episodes. Batch 4 and 1,000 steps is a starting debug budget, subject to memory and convergence. Gradient checkpointing and vision/VLM freezing may help debugging; record those choices and restore intended comparison settings.

**Prerequisite:** Verify section 5's chunk processor before training. A generic π0 command does not establish this project's action semantics. Save preprocessing and normalization with the checkpoint.

### Paper settings

| Parameter | AirVLA value |
|:--|:--|
| Initial model | π0 base |
| Optimizer / steps / global batch | AdamW / 30,000 / 32 |
| Training horizon | 50 actions |
| Learning-rate schedule | Cosine; 1,000-step warmup |
| Peak / final learning rate | $2.5\times10^{-5}$ / $2.5\times10^{-6}$ |
| Naïve deployment | 50 actions per inference, executed at 10 Hz |
| RTC | 25-action horizon; exponential prefix attention, horizon 25 |
| RTC sampling | 10 denoising steps |

Source: [AirVLA Appendix VII-A](https://arxiv.org/html/2603.25038v1#A1.SS1). Defaults may differ. Global batch includes all devices/accumulation; distinguish it from per-device batch.

### Deployment adapter

Keep one loaded policy and its processors. Collect matching features, capture the chunk anchor, generate actions using the supported API, unnormalize/decode once, and send setpoints to the controllers.

Use a process/service boundary between simulator and model environments. Replay expert targets through it before learned targets.

Keep physics and low-level control running during inference. Define queue-empty behavior, such as holding the last valid setpoint, and log each fallback. Clear queues/reject old results on reset. Record generation/execution horizons, inference latency, and simulation-time behavior.

## 8. RTC and payload guidance

### RTC

After the naïve baseline works, overlap generation and execution. Preserve the prefix committed during inference and condition the flexible suffix inside the flow sampler.

LeRobot documents [built-in RTC support](https://huggingface.co/docs/lerobot/rtc); inspect the pinned revision first. Post-hoc blending alone does not establish the same sampler behavior.

For chunk-relative actions, decode the old overlap to absolute targets, then re-encode against the new anchor before conditioning. Use the training normalization. Track timestamps/delay so overlap refers to the correct future actions.

### Payload guidance

The paper-level formulation retained from the original guide is:

$$
v_{\mathrm{guided}}=v_\theta-s(\tau)\xi,\qquad
\xi=\nabla_{x_\tau}\Phi(\hat A;o),\qquad
z_{\mathrm{des}}=z_{\mathrm{curr}}+\Delta z.
$$

Reported parameters: $\lambda_z=0.5$, $\Delta z=0.15$ m, temporal weighting $w_k=(k/(H-1))^\gamma$ with $\gamma=1$, and payload confidence from the last four gripper commands plus aperture. See the [guidance formulation](https://arxiv.org/html/2603.25038v1#S3.SS4) and [inference settings](https://arxiv.org/html/2603.25038v1#A1.SS1).

Implement the loss, terminal-action estimate, confidence heuristic, and time convention against the actual policy code. This formula is not a drop-in patch: integrators may use different flow-time directions.

Keep the path differentiable through decoding/denormalization as needed and measure altitude in physical units. Check finite gradients, intended correction direction, and equivalence to unguided sampling when guidance is zero. Preserve committed RTC actions.

Guide action generation; never directly move the simulated drone pose. Compare fixed-controller, fixed-checkpoint trials across masses, including empty-gripper cases.

## 9. Evaluation

Freeze seeds, masses, success definitions, timeouts, and controller settings before comparison.

| ID | Training data | Deployment | Question |
|:--|:--|:--|:--|
| A | Manipulation | Naïve π0 | Does learned control work? |
| B | Same as A | Same checkpoint + RTC | What changes with RTC? |
| C | Same as A | Same checkpoint + RTC + guidance | What changes with guidance? |
| D | Navigation | Naïve π0 and RTC | Navigation baseline |
| E | D's data + corrective rollouts | Naïve π0 and RTC | Do recovery examples help? |
| F | Mixed atomic-task data | RTC + guidance | Can it execute the unseen composed prompt? |

Match training budgets for D/E; disclose horizon/sampling differences. Save per-trial counts and failures, not only percentages.

Define stage criteria in configuration: object lifted and retained, object released/resting in the bin, collision-free gate crossing, and hover error/dwell time. Record thresholds before examining method comparisons.

Report:

- Pick success = successful picks / all attempts.
- Conditional place success = placements / successful picks; use N/A if no picks.
- Full-task success = completed tasks / all attempts.
- Gate/hover/composition stage counts, naming every denominator.
- Altitude sag, retention, collisions, tracking error, control effort, and latency.

### Published reference, not a simulator target

| Method | Pick | Place given successful pick |
|:--|--:|--:|
| π0 naïve | 50% | 0% |
| π0 + RTC | 85% | 23.5% |
| π0 + RTC + payload guidance | 100% | 50% |

These are real-world results from [AirVLA's benchmark table](https://airvla.github.io/). Different physics, imagery, demonstrations, and scenarios prevent direct numerical equivalence. Test whether relative improvements persist and report when they do not.

## 10. Troubleshooting

| Symptom | Check first |
|:--|:--|
| Payload has no effect | Dynamic body, wrench application, force-transmitting grasp |
| Hover/grasp fails before learning | Mass/inertia, frames/gains, contacts/friction, actuator limits |
| Images have correct shapes but wrong views | Optical convention, parent transform, camera order |
| Expert replay goes to wrong locations | Anchor, frame, yaw wrap, one-time delta conversion |
| Learned commands are too large/small | Units, normalization/statistics, padding |
| Offline fit is good but rollout fails | Image age, alignment, preprocessing, observation distribution |
| Jumps between chunks | Overlap timing, stale results, re-anchoring, queue exhaustion |
| Dataset looks better than evaluation | Scenario leakage and success-only filtering |
| Guidance moves altitude the wrong way | Flow-time direction, sign, units, gradient path |
| Guidance adds no benefit | Measured controller payload rejection and confidence gating |

The [working checklist](implement_guide.md) defines what to verify and save before the next milestone.
