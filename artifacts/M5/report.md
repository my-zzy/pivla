# M5 — Closed-loop evaluation of the M4 debug checkpoint

Status: **evaluation complete on 2026-10-10; learned policy unsuccessful**. Expert replay passed, but the unchanged M4 debug checkpoint achieved **0/2 familiar and 0/4 held-out pick-and-place successes**, with no physical grasps or lifts. The deployment and timing audit passed. No additional training, model downloads, or environment changes were performed.

Evidence: [results](result.json), [execution audit](verification.json), and [trajectory/contact plots](rollout_metrics.png). M5's measurement milestone is complete for this debug checkpoint; a useful trained control baseline is still needed before RTC comparisons.

## Closed-loop results

| Source episode | Split | Duration | Outcome | Closest distance to grasp pose | Video |
|---|---|---:|---|---:|---|
| 16, expert replay | Held-out | 51.7 s | Successful physical pick-and-place | 0.003 m | [Replay](replay/episode_016/rollout.mp4) |
| 0, π0 | Familiar | 11.1 s | Rejected out-of-bounds target | 0.584 m | [Rollout](closed_loop/trials/episode_000/rollout.mp4) |
| 4, π0 | Familiar | 80.0 s | Timeout without grasp | 0.599 m | [Rollout](closed_loop/trials/episode_004/rollout.mp4) |
| 16, π0 | Held-out | 12.2 s | Rejected out-of-bounds target | 0.542 m | [Rollout](closed_loop/trials/episode_016/rollout.mp4) |
| 17, π0 | Held-out | 80.0 s | Descended, missed grasp, moved away | 0.091 m | [Rollout](closed_loop/trials/episode_017/rollout.mp4) |
| 18, π0 | Held-out | 80.0 s | Timeout without grasp | 0.586 m | [Rollout](closed_loop/trials/episode_018/rollout.mp4) |
| 19, π0 | Held-out | 80.0 s | Descended, missed grasp, moved away | 0.207 m | [Rollout](closed_loop/trials/episode_019/rollout.mp4) |

The two rejected chunks contained a target above the 2.5 m test ceiling within their first ten actions. Those chunks were not applied. In the other trials, the drone followed drifting/incorrect targets and never established two-finger block contact. Episode 17 got closest but still missed the desired grasp pose by 9.1 cm; the block is only 8 cm wide. All payloads remained on the table. The complete trajectory/contact plots and selected approach, lift, and final frames were visually inspected; every video passed frame-count verification.

## Timing and execution audit

Across 313 accepted responses, model/preprocessing inference latency was **92.83 ms median, 95.30 ms p95**, with a 99.43 ms maximum. End-to-end latency, including IPC and polling, was 135.53 ms median and 274.77 ms p95 in wall time. Each accepted response arrived after **0.1 s of simulation and 20 physics ticks**. The statistics exclude the two rejected chunks.

Physics ran for 68,660 ticks across learned trials. All 3,118 applied policy commands matched the corresponding decoded prediction exactly, with one fixed request-time anchor per chunk. Action application remained at 10 Hz, camera age was at most 0.1 s when requested, and empty queues held the preceding absolute command. Controller computation remains scheduled every two physics ticks (100 Hz). Simulation progressed at 0.807–0.827 simulated seconds per wall second, so these runs do not establish a strict real-time deployment guarantee. Synchronous RGB rendering still pauses simulation; policy inference itself does not.

This rules out the tested conversion, ordering, stale-response, and skipped-physics faults. It does not isolate how much failure comes from model fitting, data coverage, temporal ambiguity, or the naive inference delay. Successful replay and command-tracking plots support treating policy quality as the next priority rather than changing the flight controller.

## Frozen protocol

The checkpoint is `artifacts/M4/debug_refine/checkpoint/`: an action-expert-only fit to training episodes 0 and 4. This measures a debug policy, not the proposed full-finetuning baseline. Settings were fixed in `configs/eval.yaml` before measurements.

- Familiar starts: source episodes 0 and 4. Held-out starts: 16–19, all from layout 4. These held-out starts are nearby variations, not deliberate OOD tests.
- Keep the original scene, 100 g payload, controller gains, 200 Hz physics, 100 Hz flight controller, 10 Hz action application, and 5 Hz RGB cameras.
- Predict 50 actions using ten flow-matching inference steps; execute the first ten actions. Request a new chunk only after the queue empties. This is a naive baseline, without RTC or overlapping chunk conditioning.
- Inference runs in a separate `lerobot` process. While waiting, physics/control continue with the last absolute position/yaw/gripper command. Before the first response, hold the initial observed pose with the gripper open. Decode each returned chunk against its request-time observation, never the arrival-time state.
- Best-effort real-time pacing never skips physics ticks. Rendering can make simulation slower than wall time; record both clocks and every wait. Start the returned chunk at action zero on arrival, without delay compensation. The resulting observation staleness is an explicit property of this baseline.
- Success requires a physical two-finger grasp, lifting the block more than 0.30 m above the table, then release and stable placement inside the bin for two seconds. The bin check includes the cube's conservative enclosing radius, vertical error below 0.025 m, and speed below 0.04 m/s. Trial timeout is 80 simulation seconds. Unsafe drone states or predicted targets terminate the trial.

## Offline evaluation

The unchanged checkpoint was measured on all four held-out episodes and the two familiar episodes. Sample every 20 rows within each episode, additionally including gripper transitions and episode ends. Noise seed is 7500 plus dataset row index. This differs from M4's seed and sample selection, so the familiar scores are not expected to match M4 exactly.

| Group | Evaluated chunks | Valid future actions | Position-command RMSE | Yaw RMSE | Gripper accuracy |
|---|---:|---:|---:|---:|---:|
| Familiar training episodes | 62 | 2,887 | 0.079293 m | 0.019597 rad | 97.16% |
| Held-out episodes | 124 | 5,820 | 0.087444 m | 0.012714 rad | 98.30% |

These use recorded expert observations and do not measure learned flight. Open/closed class accuracies and source-episode IDs are recorded per episode in `offline/result.json`. LeRobot numbers validation episodes 0–3 locally; they correspond to source episodes 16–19. Held-out data was not used for training or normalization.

## Interface checks

- Five π0 adapter tests passed, including exact equality of live RGB/state preprocessing and an actual recorded LeRobot sample.
- Three deployment tests passed: replay conversion over all six selected episodes, request-time anchors/queue length/stale-response rejection, and physical success criteria requiring lift, release, and dwell.
- Physical expert replay of source episode 16 succeeded: 517 commands, 10,340 physics steps, 51.7 simulated seconds. Maximum command reconstruction error was 5.96e-8. The block was lifted and released inside the bin. Initial/post-lift/final camera views and complete trajectory evidence are saved under `replay/episode_016/`.

## Reproduction

Run from the project root. Output directories must be fresh; existing recordings are never overwritten by the evaluation entry points.

```bash
bash scripts/run_lerobot.sh scripts/test_pi0_adapter.py
PYTHONPATH=airvla_isaac/source/airvla_isaac \
  /home/user/miniconda3/envs/isaaclab/bin/python -B scripts/test_m5_rollout.py
bash scripts/run_lerobot.sh scripts/m5_offline.py
bash scripts/run_isaac.sh scripts/m5_sim.py --mode replay --enable_cameras \
  --output artifacts/M5/replay
/home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m5_run.py
MPLCONFIGDIR=/home/user/pivla/artifacts/M5/plot_cache \
  /home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m5_summarize.py
```

The supervisor requires passing replay evidence, starts the offline worker, then starts headless simulation through the existing read-only launchers. Only state, RGB, and the fixed task instruction reach π0; privileged block/contact fields are used solely for outcome measurement. IPC requests use unique episode/request IDs and atomic writes, and do not deserialize Python objects.

Audit recorded execution without launching simulation:

```bash
PYTHONPATH=airvla_isaac/source/airvla_isaac \
  /home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m5_verify.py
```

Shared source and environment installations remain read-only during execution. The M3 scenario class was moved into a reusable module without changing its reset/layout behavior. Raw images, per-tick logs, inference requests, and videos stay Git-ignored; the result summary, report, and trajectory plot are retained.

## Next work

1. Establish training beyond the two-episode smoke test, initially using all 16 existing training episodes. The proposed full-finetuning configuration still needs its own memory probe; it was not run here.
2. Expand training coverage with varied starts/layouts and corrective approaches/recovery demonstrations. More nearly identical successes alone will not establish robustness. Keep the existing validation split fixed and reserve a separate OOD suite.
3. Repeat offline and closed-loop evaluation, and add controlled image/state ablations before claiming visual grounding. Neither the current offline scores nor these failures establish which inputs the model relies on.

Do not treat RTC as a fix for inaccurate commands. Defer RTC/payload comparisons until the learned policy has useful task success. These measurements concern only this checkpoint, six seeds, the stated execution schedule, and nearby layouts; they are not a general assessment of π0.
