# M5 comparison after all-16-episode training

**Neither checkpoint completed a pick-and-place in the six fixed trials.** Each produced one closed-gripper contact event, and neither achieved a lift.

Both checkpoints were evaluated with the original M5 trials, controller, timing, seeds, success criteria, and inference settings. Only the checkpoint and its descriptive scope changed. The old debug evaluation remains intact.

| Checkpoint | Familiar command RMSE | Validation command RMSE | Familiar successes | Validation successes | Closed-gripper contact / lift / place (all 6) |
|---|---:|---:|---:|---:|---:|
| Step 7,500 | 6.00 cm | 7.00 cm | 0/2 | 0/4 | 1 / 0 / 0 |
| Step 10,000 | 4.72 cm | 8.86 cm | 0/2 | 0/4 | 1 / 0 / 0 |

The historical two-episode debug checkpoint scored 0/2 familiar and 0/4 validation successes, with 8.74 cm validation command RMSE. Lower error on recorded expert observations has not established successful closed-loop control. The contact metric records any simultaneous two-finger contact while commanded closed; it does not by itself mean a retained grasp.

## Scope and training

The server run trained the action expert/projections for 10,000 optimizer steps, batch 4, on 8,188 rows from 16 demonstrations. The vision-language backbone stayed frozen. Optimizer time was 5,033 s (about 84 minutes), peak allocated GPU memory was 10.97 GiB, and checkpoint reload predictions matched exactly. No new training was run during this evaluation.

Step 7,500 was selected using the training-time validation score. The four validation layouts therefore serve as development evaluation, not an untouched test set or an OOD benchmark. M5 uses denser sampling and different noise seeds than training-time validation, so its RMSE differs from the training log. The two familiar trials cover only two of the sixteen training demonstrations. Six starts per checkpoint do not establish a general success rate.

## Trial results and videos

| Step | Source episode | Outcome | Closed-gripper contact | Lift | Closest initial grasp-pose error | Video |
|---|---:|---|---|---|---:|---|
| 7,500 | 0 | timeout | False | False | 9.6 cm | [Watch](baseline_step_007500/closed_loop/trials/episode_000/rollout.mp4) |
| 7,500 | 4 | timeout | False | False | 8.1 cm | [Watch](baseline_step_007500/closed_loop/trials/episode_004/rollout.mp4) |
| 7,500 | 16 | unsafe_state | False | False | 9.0 cm | [Watch](baseline_step_007500/closed_loop/trials/episode_016/rollout.mp4) |
| 7,500 | 17 | timeout | False | False | 8.5 cm | [Watch](baseline_step_007500/closed_loop/trials/episode_017/rollout.mp4) |
| 7,500 | 18 | unsafe_state | True | False | 1.4 cm | [Watch](baseline_step_007500/closed_loop/trials/episode_018/rollout.mp4) |
| 7,500 | 19 | timeout | False | False | 10.2 cm | [Watch](baseline_step_007500/closed_loop/trials/episode_019/rollout.mp4) |
| 10,000 | 0 | unsafe_state | True | False | 2.0 cm | [Watch](baseline_step_010000/closed_loop/trials/episode_000/rollout.mp4) |
| 10,000 | 4 | timeout | False | False | 4.0 cm | [Watch](baseline_step_010000/closed_loop/trials/episode_004/rollout.mp4) |
| 10,000 | 16 | timeout | False | False | 54.2 cm | [Watch](baseline_step_010000/closed_loop/trials/episode_016/rollout.mp4) |
| 10,000 | 17 | timeout | False | False | 53.6 cm | [Watch](baseline_step_010000/closed_loop/trials/episode_017/rollout.mp4) |
| 10,000 | 18 | timeout | False | False | 60.0 cm | [Watch](baseline_step_010000/closed_loop/trials/episode_018/rollout.mp4) |
| 10,000 | 19 | timeout | False | False | 57.7 cm | [Watch](baseline_step_010000/closed_loop/trials/episode_019/rollout.mp4) |

Closest-pose error uses the initial block position and configured drone grasp height. Once the block is displaced, this is not the instantaneous finger-to-block alignment error.

Trajectory plots: [step 7,500](baseline_step_007500/rollout_metrics.png), [step 10,000](baseline_step_010000/rollout_metrics.png).

## Verification

- Both downloaded models and all preprocessing/tokenizer files match the server SHA256 manifest. Matching local weight blocks were reused to reduce transfer time; the reconstructed files are byte-identical to the server checkpoints.
- A fresh expert replay passed: 517 commands, 10,340 physics ticks, 51.7 simulated seconds, maximum command reconstruction error 5.96e-8.
- Five adapter and three deployment tests passed.
- Each evaluation audits every applied command against its request-time decoded prediction, camera age, queue holds, and physics progression during inference. The audit also checks every logged physics tick, including runs stopped partway through an action interval.
- Video frame counts were verified. The existing environments and shared source checkouts were preserved; no packages were installed.

Detailed machine-readable evidence: [comparison](baseline_comparison.json). Configurations and raw logs are under `baseline_step_007500/` and `baseline_step_010000/`; large recordings and model files remain Git-ignored.

## Reproduce

Run from the repository root with downloaded checkpoints present. Use fresh output directories; evaluation entry points reject existing output directories. For each saved `eval.yaml`:

```bash
root=artifacts/M5/baseline_step_007500  # or baseline_step_010000
out=artifacts/M5/recheck_007500        # choose a fresh directory
bash scripts/run_lerobot.sh scripts/m5_offline.py --config "$root/eval.yaml" --output "$out/offline"
/home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m5_run.py \
  --config "$root/eval.yaml" --replay-root artifacts/M5/baseline_replay --output "$out/closed_loop"
bash scripts/run_lerobot.sh scripts/m5_verify.py \
  --config "$root/eval.yaml" --replay-root artifacts/M5/baseline_replay --root "$out"
MPLCONFIGDIR="$PWD/artifacts/M5/plot_cache" /home/user/miniconda3/envs/isaaclab/bin/python -B \
  scripts/m5_summarize.py --root "$out" --policy-only
```

## Observed behavior and next work

The step-7,500 policy descended toward the block in several trials, but missed alignment or failed to retain contact. Validation episodes 16 and 18 stopped below the safe height; the other four trials timed out. The final checkpoint stopped below the safe height in familiar episode 0, displaced the block off the table in familiar episode 4, and remained near hover height with an open gripper throughout all four validation trials. These descriptions follow the recorded trajectories and camera checks; they do not isolate a single underlying cause.

Both execution audits passed. This excludes the tested command-decoding, anchor, ordering and skipped-physics faults; it does not prove that controller/contact dynamics, policy timing, visual dependence or dataset coverage are sufficient. Improved offline fitting alone did not resolve the closed-loop failures.

Next, inspect the hover-to-approach transition and grasp alignment, run controlled RGB/state ablations, and design corrective demonstrations from the observed off-demonstration states. Keep step 7,500 as a diagnostic reference because its validation predictions are better, but do not call either checkpoint a successful controller. Reserve a new untouched/OOD test suite. Defer RTC until the baseline achieves useful task success; no new training or data collection was started here.
