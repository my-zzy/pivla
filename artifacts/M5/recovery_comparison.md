# M5 evaluation after recovery training

**Neither recovery checkpoint completed a fixed M5 pick-and-place trial.** Step 2,500 and step 10,000 each scored 0/2 familiar and 0/4 validation successes, with no closed-gripper contact or lifts. All twelve trials climbed away from the block and stopped when the next predicted command exceeded the 2.5 m height limit. The rejected command was not applied.

| Checkpoint | Familiar command RMSE | Validation command RMSE | Familiar successes | Validation successes | Contact / lift / place (all 6) |
|---|---:|---:|---:|---:|---:|
| Previous baseline step 7,500 | 6.00 cm | 7.00 cm | 0/2 | 0/4 | 1 / 0 / 0 |
| Previous baseline step 10,000 | 4.72 cm | 8.86 cm | 0/2 | 0/4 | 1 / 0 / 0 |
| Recovery step 2,500 | 17.87 cm | 16.42 cm | 0/2 | 0/4 | 0 / 0 / 0 |
| Recovery step 10,000 | 17.94 cm | 16.82 cm | 0/2 | 0/4 | 0 / 0 / 0 |

“Familiar” refers to the original training layouts; these exact M3 trajectories were not part of the replacement recovery dataset. The comparison uses the original M3 recordings and the same six M5 starts, seeds, controller, success criteria, inference steps and action schedule. Offline inputs and targets use each checkpoint's saved training normalization. Only dataset provenance metadata may differ; feature semantics must match. The original timed demonstration targets remain an imperfect measure of the state-driven behavior we want, so physical outcomes are the deciding evidence.

## Training and checkpoint selection

The server run completed 10,000 optimizer steps at batch 4 on 6,534 rows from 18 training episodes. The vision-language backbone stayed frozen. Optimizer time was 5,024.5 s (83.7 minutes), peak allocated GPU memory was 10.97 GiB, and checkpoint reload predictions matched exactly. Final training command RMSE was 2.21 cm; training-time validation RMSE was 15.93 cm.

Step 2,500 was selected before simulation because it had the lowest recorded validation command RMSE (14.93 cm) among the saved trained checkpoints. The final checkpoint was also evaluated. Validation therefore remains development data, not an untouched or deliberate OOD test. Offline scores here use the denser M5 sampling and differ from the training log. No new training was launched during this evaluation.

## Trial results and videos

| Step | Source episode | Outcome | Simulated duration | Video |
|---|---:|---|---:|---|
| 2,500 | 0 | unsafe_command | 22.1 s | [Watch](recovery_step_002500/closed_loop/trials/episode_000/rollout.mp4) |
| 2,500 | 4 | unsafe_command | 33.1 s | [Watch](recovery_step_002500/closed_loop/trials/episode_004/rollout.mp4) |
| 2,500 | 16 | unsafe_command | 22.1 s | [Watch](recovery_step_002500/closed_loop/trials/episode_016/rollout.mp4) |
| 2,500 | 17 | unsafe_command | 27.6 s | [Watch](recovery_step_002500/closed_loop/trials/episode_017/rollout.mp4) |
| 2,500 | 18 | unsafe_command | 33.1 s | [Watch](recovery_step_002500/closed_loop/trials/episode_018/rollout.mp4) |
| 2,500 | 19 | unsafe_command | 25.4 s | [Watch](recovery_step_002500/closed_loop/trials/episode_019/rollout.mp4) |
| 10,000 | 0 | unsafe_command | 16.6 s | [Watch](recovery_step_010000/closed_loop/trials/episode_000/rollout.mp4) |
| 10,000 | 4 | unsafe_command | 19.9 s | [Watch](recovery_step_010000/closed_loop/trials/episode_004/rollout.mp4) |
| 10,000 | 16 | unsafe_command | 18.8 s | [Watch](recovery_step_010000/closed_loop/trials/episode_016/rollout.mp4) |
| 10,000 | 17 | unsafe_command | 19.9 s | [Watch](recovery_step_010000/closed_loop/trials/episode_017/rollout.mp4) |
| 10,000 | 18 | unsafe_command | 19.9 s | [Watch](recovery_step_010000/closed_loop/trials/episode_018/rollout.mp4) |
| 10,000 | 19 | unsafe_command | 17.7 s | [Watch](recovery_step_010000/closed_loop/trials/episode_019/rollout.mp4) |

Trajectory plots: [step 2,500](recovery_step_002500/rollout_metrics.png), [step 10,000](recovery_step_010000/rollout_metrics.png).

## Heading coverage and separate diagnostic

The recovery collector explicitly reset initial yaw to zero, and this dataset replaced the original training data. Its observed yaw range is only −0.00515 to +0.00387 rad (about −0.30° to +0.22°), with standard deviation 0.00121 rad. All six original evaluation starts lie outside this range: their normalized yaw magnitudes are 41.9–122.4. A valid dataset format and a successful expert replay did not establish adequate state coverage.

After the fixed benchmark, a separate diagnostic used the final checkpoint on familiar episodes 0 and 4 with only initial yaw set to zero. All controller settings, position starts, inference seeds and safety checks stayed the same. These trials are excluded from the benchmark totals.

| Diagnostic episode | Outcome | Contact | Lift | Closest initial grasp-pose error | Video |
|---|---|---|---|---:|---|
| 0 | timeout | False | False | 9.6 cm | [Watch](recovery_yaw_zero_diagnostic/closed_loop/trials/episode_000/rollout.mp4) |
| 4 | unsafe_state | True | False | 0.7 cm | [Watch](recovery_yaw_zero_diagnostic/closed_loop/trials/episode_004/rollout.mp4) |

With zero initial yaw, both trials approached the block. Episode 0 later climbed without the block and timed out; episode 4 registered closed-gripper contact, then crossed the lower safe-height limit. Neither lifted successfully. This supports heading sensitivity as a contributor, while also showing that heading alone does not explain the grasp/control failures. Changing physical yaw also changes the rendered observations; the diagnostic does not isolate the numeric state input from image effects.

## Verification

- Both downloaded checkpoints, tokenizer files and preprocessing metadata matched the server SHA256 manifest: 16 files, 17,853,796,531 bytes. Matching local weight blocks were reused during transfer.
- Fresh expert replay passed: 517 commands, 10,340 physics ticks, 51.7 simulated seconds, maximum command reconstruction error 5.96e-8.
- Six adapter tests and three rollout tests passed. The added test checks that original recorded states match live preprocessing under recovery statistics, absolute action targets round-trip correctly, and incompatible frame semantics are rejected.
- Both fixed evaluations and the diagnostic passed the execution audit, including request-time anchors, decoded applied commands, camera ages, queue holds and uninterrupted physics progression.
- All fourteen rollout videos passed frame-count checks. Representative frames were inspected. Existing environments and shared source checkouts were preserved; no packages were installed.

The audits exclude the checked transfer, preprocessing and action-execution faults. They do not prove sufficient data coverage, visual grounding, or robust contact control. Full settings, checkpoint hashes, metrics and diagnostic evidence are in [recovery_comparison.json](recovery_comparison.json). Large models, videos and raw logs remain Git-ignored.

## Next work

Keep the state-driven expert, but collect it from the original training start distribution, including the existing yaw range, and retain the recovery segments. Check state coverage before training; validate approach, failed grasp and loss-of-contact recovery. Then run a small learning pilot and repeat the unchanged physical evaluation before another full training budget. Do not change normalization on an already-trained checkpoint or weaken the benchmark. RTC and a fresh untouched/OOD suite remain later work.

## Reproduce

Run from the repository root with downloaded models present and a fresh output directory. Exact configurations are saved in each local evaluation directory and in the tracked comparison JSON.

```bash
root=artifacts/M5/recovery_step_002500  # or recovery_step_010000
out=artifacts/M5/recovery_recheck_002500
bash scripts/run_lerobot.sh scripts/m5_offline.py --config "$root/eval.yaml" --output "$out/offline"
/home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m5_run.py \
  --config "$root/eval.yaml" --replay-root artifacts/M5/recovery_replay --output "$out/closed_loop"
bash scripts/run_lerobot.sh scripts/m5_verify.py \
  --config "$root/eval.yaml" --replay-root artifacts/M5/recovery_replay --root "$out"
MPLCONFIGDIR="$PWD/artifacts/M5/plot_cache" /home/user/miniconda3/envs/isaaclab/bin/python -B \
  scripts/m5_summarize.py --root "$out" --policy-only
```
