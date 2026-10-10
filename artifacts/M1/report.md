# M1 — Force-driven hover and three cameras

**Passed on 2026-10-10. Next: M2, physical grasp-and-lift.**

The external package now contains a procedural room, a gravity-enabled 1 kg drone, a geometric position/yaw controller, and three RGB cameras. The standalone M1 runner is the main simulation entry point. The generated M0 cartpole task remains available as an installation smoke test.

## Results

The drone started 0.493 m from its target with a 0.20 rad yaw offset. After a fixed 5 s settling period, it held the target for 20 s. The 0.10 m position threshold was specified before the run.

| Check | Measured result |
|:--|:--|
| Physics run | 5,000 steps at 200 Hz; 25 simulated seconds |
| Hover window | 4,000 samples, `5 < t <= 25 s` |
| Maximum hover position error | **0.003684 m** (limit: 0.10 m) |
| RMS hover position error | 0.000711 m |
| Maximum hover yaw error | 0.00000570 rad (limit: 0.05 rad) |
| Mean hover thrust | 9.810003 N; configured weight: 9.81 N |
| Three same-seed resets | Identical initial states and identical one-second controlled trajectories |
| Different seed | Changes the configured initial-position jitter |
| Unpowered 0.15 s fall | 0.1140 m drop; vertical speed −1.471500 m/s, matching gravity |
| Mass and inertia | Verified through the PhysX body view |
| Cameras | 125 frames per stream; 256 × 256 uint8 RGB at 5 Hz |
| Maximum camera-position error | 0.000000135 m against the fixed extrinsic transform |
| Camera rotation and timestamps | Passed for every capture |
| Desktop GUI regression | Full 25 s run passed; exactly 5,000 physics-engine ticks, maximum hover error 0.003684 m |
| Controller checks | Four CPU tests passed: equilibrium, restoring signs, saturation, frame rotation |

The first, middle, and final camera previews were inspected. The front view sees the orange/blue gate, the downward view sees the green pad and yellow +X marking, and the external view sees the drone and room. The front sensor is ahead of the rotor geometry so its view is unobstructed. The encoded H.264 video was checked: 768 × 296, 125 frames, 5 fps, 25 s.

- [Three-camera preview](camera_preview.png)
- [Altitude, error, thrust and torque plots](hover_plots.png)
- [Hover video](hover.mp4) — generated locally, ignored by Git
- [Machine-readable results](result.json), [reset and gravity checks](reset_checks.json)
- [Desktop GUI validation](gui_validation.json)
- [Environment preservation](environment_preservation.json)

Raw `flight.jsonl`, `cameras.jsonl`, per-camera PNGs, the resolved `config.json`, and `hover.log` remain in this directory and are ignored by Git. Flight logs contain pose, velocity, angular velocity, target pose, body thrust/torque, simulation time and command time. Camera logs contain simulation capture/availability times, wall-clock availability, intrinsics and measured world poses.

## Reproduce

From `/home/user/pivla`:

The run below was verified under the agent's VS Code AppArmor context. A subsequent ordinary-terminal attempt hit `bwrap: setting up uid map: Permission denied`; see the [confirmed cause and one-time setup](../../docs/launcher-troubleshooting.md). This permission issue occurs before the simulator starts and does not change the recorded physics results.

```bash
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras

MPLCONFIGDIR=/home/user/pivla/artifacts/M1/plot_cache \
  /home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m1_summarize.py

/home/user/miniconda3/envs/isaaclab/bin/python -B scripts/test_flight_controller.py
```

For a two-second debugging run, add `--smoke --output artifacts/M1/debug` to the simulator command. A smoke run never certifies M1. Full results require `result.json` to say `passed` and every check to be true.

## Physics and timing contract

All settings are in [configs/sim.yaml](../../configs/sim.yaml). World and body frames are right-handed: world +Z up; body +X forward, +Y left, +Z up. Quaternions are `wxyz` and map body vectors into world coordinates. Distances, mass, forces, torque and angles use metres, kilograms, newtons, newton-metres and radians.

The body has mass 1 kg, centre of mass at its origin, and diagonal inertia `[0.012, 0.015, 0.025] kg m²`. Body/arms have simple collision geometry. The rotor discs are visual only. Gravity is `[0, 0, -9.81] m/s²`; linear and angular damping are zero. Root pose/velocity writes happen only in `reset()`.

The controller runs at 100 Hz. Position gains are `[4, 4, 6]`, velocity gains `[3.5, 3.5, 4.5]`, attitude gains `[0.6, 0.6, 0.4]`, and angular-velocity gains `[0.16, 0.18, 0.16]`. Desired attitude aligns body +Z with the desired acceleration and respects the yaw target. Desired angular-rate feedforward is zero for this hover/slow-setpoint baseline. Torque includes the gyroscopic term. Tilt is limited to 0.52 rad, thrust to `[0, 22] N`, and each torque component to `±[1, 1, 0.5] N m`.

At a controller tick, `permanent_wrench_composer.set_forces_and_torques(..., is_global=False)` replaces the body-frame wrench. `write_data_to_sim()` applies it before every 200 Hz physics step. Force acts at the body centre of mass, along body +Z; the wrench persists between controller updates. No flight state is teleported to track a target.

The GUI path uses the same `step(render=False)` call as headless mode, followed by a separate render-only refresh at 50 Hz. Camera recording remains 5 Hz. A regression in the first GUI implementation used `step(render=True)`, which let the installed core API advance a complete 0.2 s render interval while the controller counted one 0.005 s tick. This caused the failed free-fall check and rapid fall reported from the desktop run. The corrected environment asserts that every physics call advances exactly one engine tick and that warm-up, camera capture and viewport-only rendering advance none. Failed preflight checks now stop before hover, and exceptions mark the result as `failed` rather than leaving it `running`.

After the fix, both full GUI and headless runs passed all checks with exactly 5,000 engine ticks. Their recorded positions, orientations, velocities, angular velocities and applied wrenches matched exactly across all 5,000 samples (maximum absolute difference: 0). The main video/plots and result files were regenerated from the corrected headless run.

Front and down sensors have fixed body extrinsics. Their world-parented camera prims are synchronized from the actual PhysX body pose immediately before capture, using the public camera API. This avoids stale parent transforms in the pinned Isaac Lab USD/Fabric camera path without modifying shared source. Only camera prims are repositioned. Every capture checks the actual sensor pose against the expected rigid mount. The external camera stays fixed.

Cameras use the Isaac Lab `world` convention (+X forward, +Z up); the API converts it to USD camera axes. Front offset is `[0.43, 0, -0.02] m`. Down offset is `[0, 0, -0.09] m`, rotated +90° around body Y. Intrinsics are `fx=fy=192 px`, `cx=cy=128 px`. Frames are captured every 40 physics steps. Rendering is synchronous while physics is paused, so capture and availability simulation times coincide; wall-clock availability is recorded separately. Startup rendering produces no dataset samples.

## Scope and environment preservation

This verifies unloaded hover in a deterministic, simplified scene. It does not establish motor-level fidelity, payload handling, robust navigation, contact manipulation, or learned-policy performance. The next milestone adds a physical gripper and object contact before evaluating loaded flight.

Both shared environments were reused without installation or upgrades. The launcher mounts shared installations read-only and stores M1 runtime files under `artifacts/M1/runtime/`. Before/after fingerprints matched for 3,497 simulator metadata files and 1,276 LeRobot metadata files; all three shared source checkout commits and Git statuses were unchanged. Video, raw images/logs, debugging runs and caches stay ignored; only small M1 evidence is eligible for commit.
