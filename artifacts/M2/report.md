# M2 — Physical scripted pick-and-place

Status: **passed on 2026-10-10 — 10/10 complete physical pick-and-place trials in the fixed scene.**

Watch the local [51.4-second video](pick_place.mp4) without starting the simulator. See the [camera preview](camera_preview.png), [load/contact plots](load_contact_plots.png), and [per-trial result](result.json).

## Measured results

| Measurement | Result |
|:--|:--|
| Successful scripted trials | 10/10 |
| Duration / physics ticks per trial | 51.4 s / 10,280 |
| Three-camera recording for trial 0 | 257 frames per stream, 256×256 RGB at 5 Hz |
| Same-seed initial-state maximum difference | 0 |
| Mean empty / loaded hover thrust | 10.300500 / 11.281521 N |
| Measured extra thrust / payload weight | 0.981021 / 0.981000 N |
| Mean empty / loaded hover altitude | 1.400001 / 1.243905 m (target: 1.40 m) |
| Maximum empty / loaded position error | 0.000478 / 0.156709 m |
| Loaded measurement with both fingers contacting block | 100% |
| Maximum camera position / rotation-matrix element error | 8.96e-8 m / 4.18e-7 |

The loaded PD equilibrium is about **15.6 cm below the target** because the controller keeps its empty-mass model and has no integral term. It still clears the table and bin walls. This is measured payload response with the original flight gains, not an unloaded-hover tracking claim. The block is supported only by finger contact while carried, and settles at the bin floor after opening.

All ten trials passed every physical check. The recorded image paths, 0.2 s camera spacing, capture/availability timestamps, and held 10 Hz commands were also checked against the raw traces. The encoded video is H.264, 768×296, 5 fps, 257 frames. Grasp, carry, release, and final frames and the plotted traces were visually reviewed.

The full [M1 regression](m1_regression/result.json) also passed all 14 checks: 5,000 engine ticks, 20-second hover window, maximum error 0.003684 m, repeatable resets/trajectories, and 125 frames per camera. Four controller unit checks and four expert transition/timeout checks passed. All ten M2 raw traces reproduced their saved hover metrics using the final integer-window calculation.

[Environment preservation](environment_preservation.json) confirms unchanged package metadata for both `isaaclab` and `lerobot`, and unchanged commits/status for both Isaac Lab checkouts and LeRobot. Nothing was installed or upgraded.

## Implementation

- The original 1 kg drone carries two 25 g fingers. The fixed mount is part of the original body's mass/inertia model; the moving fingers have their own gravity-enabled rigid bodies and computed cuboid inertias.
- Two prismatic joints constrain finger motion along body Y. Force drives use 600 N/m stiffness, 8 N s/m damping, and an 8 N maximum per finger. Aperture is 0.14 m open and 0.03 m at the closed mechanical limit; the 0.08 m block stops closure through contact.
- Fingers and block use static/dynamic friction 1.0 and zero restitution. There is no attachment joint to the block. Finger contact/friction transmits the payload to the drone through the sliding joints. Runtime root-pose/velocity writes occur only during reset.
- The flight gains, thrust limit, and torque limits are inherited unchanged from M1. Empty-mass feedforward accounts for the two fingers (1.05 kg total), and stays fixed when grasping the 0.10 kg payload. The model retains M1's base-body inertia for the controller.
- Physics runs at 200 Hz, flight control at 100 Hz, expert setpoints at 10 Hz, and recorded RGB at 5 Hz. Setpoints move at up to 0.12 m/s. Rendering does not advance physics.
- The privileged-state expert executes approach → descend → grasp → lift → loaded hold → transport → release → retreat. Pose, velocity, aperture, and filtered block/finger contact gate transitions; each stage times out after 30 s. Bin success requires the released block to settle inside the bin and remain there while the drone retreats.
- The fixed scene has a table at X=-0.65 m and bin at X=+0.65 m. Both support surfaces are at Z=0.50 m. The cube is 0.08 m on each side. Resets use seed 7 and ±0.005 m initial drone position jitter.
- The downward camera is moved forward and tilted toward the fingers to see past the mounting rail; the fixed external camera is closer to the work area. M1 camera settings remain unchanged.

The joint drive follows the [OpenUSD force-limited spring drive API](https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html). Installed Isaac Lab source was used to verify the filtered contact-sensor interface.

## Acceptance criteria recorded before validation

- Ten successful full scripted runs from the same seeded reset in the fixed scene.
- Exactly one engine physics tick per environment step; finite states and wrenches.
- Empty hover maximum position error below 0.03 m; loaded hover below 0.20 m with unchanged gains. Compare the final four seconds of each measurement stage (800 physics samples).
- Loaded-minus-empty mean thrust within 0.20 N of the payload weight (0.981 N).
- Both fingers report block contact for more than 98% of the loaded measurement window. The held block center remains at least 0.30 m above the tabletop during loaded hold and transport.
- Actual PhysX masses match the configuration. Camera frames are nonuniform 256×256 uint8 RGB; mount transforms agree with physical pose within 0.0001 m and 0.0001 rotation-matrix element error.
- The block is released, settles in the bin, and remains there during retreat. Record a complete three-camera episode and save the raw state, commands, contact measurements, and timestamps.

## Commands

From `/home/user/pivla`:

```bash
# Full validation; records cameras for trial 0 and physical traces for all 10.
bash scripts/run_isaac.sh scripts/m2_pick_place.py --enable_cameras --trials 10

# One complete recorded demo in a separate output folder.
bash scripts/run_isaac.sh scripts/m2_pick_place.py --enable_cameras --trials 1 --output artifacts/M2/demo

# Faster physics-only grasp/lift debugging; not a complete M2 validation.
bash scripts/run_isaac.sh scripts/m2_pick_place.py --trials 1 --lift-only --output artifacts/M2/lift_debug

# Create plots/video from existing data; does not launch the simulator.
MPLCONFIGDIR=/home/user/pivla/artifacts/M2/plot_cache \
  /home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m2_summarize.py

/home/user/miniconda3/envs/isaaclab/bin/python -B scripts/test_flight_controller.py
/home/user/miniconda3/envs/isaaclab/bin/python -B scripts/test_expert_policy.py

# M1 regression in a separate output folder; preserves the original M1 evidence.
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras --output artifacts/M2/m1_regression
```

The launcher uses the existing `isaaclab` environment with shared installations mounted read-only and runtime files under `artifacts/M2/runtime/`. No extra packages or asset downloads are needed. Add `--gui` only for live debugging; ordinary runs are headless.

## Evidence layout and limits

`result.json` contains per-trial checks, actual masses, stage outcomes, and load measurements. Each `episode_NNN/` has `flight.jsonl` at 200 Hz and a result; trial 0 also has three PNG image streams, mosaics, and `cameras.jsonl` with calibration/poses and capture/availability times. Simulation pauses during rendering, so capture and availability simulation timestamps coincide. `action_time_s` and `control_time_s` identify the held commands applied before each logged physics step. A failed stage writes `failure.json`.

Raw episodes, videos, debug runs, and runtime caches are ignored by Git. Only the compact result, report, environment-preservation evidence, preview, and plot are eligible for commit.

This is a procedural fixed-scene demonstration with privileged-state control, direct resultant thrust/torque, and simplified gripper geometry. Ten same-seed successes do not establish robustness to new poses, payloads, friction, or real hardware. LeRobot export, randomized demonstrations, and learned policies belong to later milestones.
