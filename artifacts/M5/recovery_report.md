# State driven expert and corrective dataset

**The new expert passed all five physical pilot cases and all sixteen collection episodes.** The separate LeRobot dataset contains eighteen training episodes, including two fault-free recovery suffixes, and the four original validation episodes. Dataset validation passed. This establishes expert/data readiness; the learned policy has not been retrained or reevaluated on this dataset.

## Expert behavior

The separate `RecoveryExpert` removes the eight-second approach wait and the measurement-only lift/loaded holds. It uses the measured block position for alignment, short 0.5-second condition dwells, a two-second stable placement check, and at most three retries. Missing closure or sustained loss of contact triggers opening, vertical retraction, realignment and another approach. A block leaving the table or an unsafe drone state stops collection. The original expert, flight controller, simulation configuration and fixed learned evaluation protocol are unchanged.

Privileged block/contact measurements are used by the expert and outcome checks. They are excluded from policy observations, which remain three RGB images, eight state values and the fixed instruction.

## Physical pilot

| Case | Duration | Retries | Result | Video |
|---|---:|---:|---|---|
| aligned | 34.3 s | 0 | passed | [Watch](recovery_pilot/episode_000/rollout.mp4) |
| lateral | 37.1 s | 0 | passed | [Watch](recovery_pilot/episode_001/rollout.mp4) |
| low_approach | 38.7 s | 0 | passed | [Watch](recovery_pilot/episode_002/rollout.mp4) |
| missed_closure | 49.2 s | 1 | passed | [Watch](recovery_pilot/episode_003/rollout.mp4) |
| lost_contact | 48.1 s | 1 | passed | [Watch](recovery_pilot/episode_004/rollout.mp4) |

Fault tests hold the actual gripper open for 2.5 seconds when closure or lift begins. The lost-contact test exercises loss of grip at lift onset, not recovery from a drop at transport height. All five videos passed frame-count checks. New and original expert unit suites each passed five tests.

## Dataset contents

| Split | Episodes | Rows |
|---|---:|---:|
| train | 18 | 6534 |
| validation | 4 | 2068 |

Training uses the four original training layouts (source episodes 0, 4, 8 and 12), each with aligned, lateral-offset, low-height and shifted-block starts. The lateral offset is +16 cm / −12 cm; low starts are at 1.04 m with −13 cm / +10 cm offsets; shifted-block cases move the block +7 cm in X. Starts use zero commanded yaw; broader yaw variation is not yet covered.

Two extra training episodes contain the successful recovery portions after each injected fault ends. No fault-active samples or their prefixes are exported. Timestamps and physics traces are rebased consistently; RGB paths point to the original lossless recordings. These suffixes are recorded demonstrations, not independently resettable simulator scenarios: exact replay requires restoration of their full starting physical state.

Validation remains the original M3 source episodes 16–19 on layout 4. None enters training normalization or corrective collection. These validation targets still contain the old timed holds, so offline prediction error alone should not decide whether the new policy improved. Use the fixed M5 physical trials for the next comparison. A fresh untouched/OOD suite remains future work.

Dataset: `datasets/airvla_m5_recovery/`. Export view and provenance: `artifacts/M5/recovery_export/`. Raw collection: `artifacts/M5/recovery_collection/`. Original datasets and checkpoints were preserved. The new normalization must travel with the new dataset/checkpoint; do not attach the old checkpoint adapter to it.

Validation checked 8,602 raw chunks, 226 native loader chunks and 219 exact image comparisons. All 13 checks passed, including observation-before-action timing, camera availability, scenario isolation, training-only statistics, padding and action reconstruction.

## Learning run and follow-up

Use the prepared configuration, which retains batch 4, a frozen vision-language backbone and the 10,000-step baseline budget. It starts from `pi0_base` with the new dataset statistics.

The disposable forward/backward/optimizer probe passed on 2026-10-21 on server port 30, GPU 0 (NVIDIA RTX PRO 5000 72GB Blackwell): 25 steps, 0.475 s/step after warmup, 10.97 GiB peak allocated memory. All 17 transferred dataset files matched their local SHA256 checksums, and train/validation samples loaded successfully. The 79-minute estimate for 10,000 steps excludes startup, validation and checkpoint saves. See `server_readiness` in [recovery_result.json](recovery_result.json).

For future server probes, activate the server environment and use the server launcher:

```bash
conda activate lerobot
LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=0 bash scripts/run_lerobot_server.sh scripts/m4_train.py \
  --config configs/train_recovery.yaml --probe --probe-steps 25
```

Check GPU availability before launching and use a fresh output directory when repeating a probe. The desktop launcher `run_lerobot.sh` uses local paths and Bubblewrap; it now gives server guidance when the desktop installation is absent. No environment packages were changed for this fix.

**Update, 2026-10-21:** The user completed full recovery training: 10,000 steps, batch 4, 83.7 minutes optimizer time, exact checkpoint reload. Both the best recorded validation checkpoint (step 2,500) and final checkpoint scored 0/6 on fixed M5 trials, with no contacts or lifts. A separate two-trial zero-yaw diagnostic improved approach and produced one contact, but no lifts or successes. See the [comparison and videos](recovery_comparison.md).

The recovery collector resets initial yaw to zero, leaving the fixed evaluation starts far outside the new training yaw distribution. The next collection should preserve the original training start/yaw distribution while using the state-driven expert and retaining recovery examples. Validate grasp/recovery and feature coverage before another small learning pilot; another full run is not yet justified.

## Reproduction

These commands select the historical v1 reset plan and require a passing five-case pilot. The current expert includes later fixes; the preserved artifacts above record the original run. For the current protocol, use the [v2 report](recovery_v2_report.md). Run from the repository root with fresh output directories.

```bash
bash scripts/run_isaac.sh scripts/m5_collect_recovery.py --protocol v1 --mode pilot --enable_cameras \
  --output artifacts/M5/recovery_pilot
bash scripts/run_isaac.sh scripts/m5_collect_recovery.py --protocol v1 --mode collect --enable_cameras \
  --pilot-result artifacts/M5/recovery_pilot/result.json --output artifacts/M5/recovery_collection
python scripts/m5_prepare_recovery_export.py --collection artifacts/M5/recovery_collection \
  --pilot artifacts/M5/recovery_pilot --output artifacts/M5/recovery_export
bash scripts/run_lerobot.sh -m airvla_isaac.data.convert_to_lerobot \
  --raw artifacts/M5/recovery_export --destination datasets/airvla_m5_recovery
bash scripts/run_lerobot.sh -m airvla_isaac.data.validate_dataset \
  --raw artifacts/M5/recovery_export --dataset datasets/airvla_m5_recovery
```

[Machine-readable checks and provenance](recovery_result.json). No environment packages or shared source files were changed.

**Follow-up, 2026-10-22:** [Recovery v2](recovery_v2_report.md) preserves original reset/yaw coverage and fixes expert lift drift. Expert/data checks passed; the 2,500-step learned pilot still scored 0/6. The remaining observed gap is precise grasp alignment and closed-empty gripper recovery.
