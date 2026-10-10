# M3 — Dataset and action contract

Status: **passed on 2026-10-10**. Collected 20/20 successful episodes, exported separate local LeRobot training/validation datasets, and passed all 13 dataset checks. No policy training has started.

## Results

| Check | Measured result |
|---|---|
| Successful demonstrations | 20/20 across five layouts |
| Train / validation | 16 episodes / 4 episodes; 8,188 / 2,068 rows |
| Absolute → relative → absolute | All 10,256 chunk starts checked; maximum error 5.96e-8 |
| Native LeRobot chunk reader | 220 checks at episode boundaries and stage changes |
| Lossless image comparisons | 165 passed across all three cameras |
| Camera timing | No future images; maximum image age 0.10 s |
| Batches | State `[4,8]`, action `[4,50,7]`, mask `[4,50]`, each image `[4,3,256,256]` |
| Contract unit checks | Four codec, one statistics, and five expert checks passed |
| Shared installations | Package metadata, editable links, source commits/status unchanged |

Evidence: [collection](collection_result.json), [export](export_result.json), [validation](validation_report.json), [split manifest](split_manifest.json), [normalization](normalization.json), and [environment preservation](environment_preservation.json).

Five full three-camera videos are saved locally: [episode 0](previews/episode_000.mp4), [4](previews/episode_004.mp4), [8](previews/episode_008.mp4), [12](previews/episode_012.mp4), and [16](previews/episode_016.mp4). Every chronological down/external frame was inspected in contact sheets, covering approach, grasp, lift, transport, release, and retreat. All five show the block remaining gripped until release and settling inside the bin. The full [action reconstruction plot](action_reconstruction.png) shows decoded positions and gripper commands overlapping their recorded targets. [Preview evidence](preview_manifest.json) records the review and video integrity checks. Video playback does not start Isaac Sim.

## Frozen contract

The authoritative configuration is [configs/dataset.yaml](../../configs/dataset.yaml). Each 10 Hz row has three 256×256 RGB images, state `[x,y,z,yaw,vx,vy,vz,aperture]`, absolute desired action `[x_des,y_des,z_des,0,0,yaw_des,gripper_closed]`, and the instruction “Pick up the block from the table and place it in the bin.” Positions/velocities use world axes and the drone body origin; units are SI, quaternion order is wxyz, and yaw wraps to `[-pi,pi)`. Gripper is 0=open, 1=closed.

The raw recorder reads state and available images first, queries the expert second, and then advances physics under that command. This avoids turning M2's post-step engineering logs into incorrectly aligned training rows. Cameras capture at 5 Hz; the intervening 10 Hz row reuses the latest available frame with its original capture/availability time. Rendering is synchronous and pauses physics. An initial warm-up capture at t=0 makes the first observation valid.

Absolute actions remain on disk. `ChunkDataset` uses LeRobot's episode-aware future-action query for a 50-step horizon, then subtracts the **same observed position/yaw anchor from every valid future target**. Gripper remains absolute, roll/pitch placeholders remain zero, yaw differences wrap, and unavailable future actions are zeroed using the padding mask. Normalization occurs after relative conversion; padded values remain zero after normalization. Decoding unnormalizes once and adds the saved anchor once.

## Collection and splits

Twenty completed episodes: five seeded block/bin layouts, four independently seeded drone resets per layout. Block XY offsets are within ±0.07 m of the M2 location; bin offsets are within ±0.08/0.10 m; nominal drone XY/Z offsets within ±0.08/0.03 m and initial yaw within ±0.15 rad. The inherited ±0.005 m reset jitter is then added to the drone position; the actual state is recorded at t=0. The table remains fixed, and all block positions lie on it. Payload mass is fixed at 0.10 kg for this debugging dataset. Flight gains and actuator limits remain the M2 baseline.

Scenarios 0–3 supply 16 training episodes; scenario 4 supplies four validation episodes. This assignment is fixed before exporting or computing statistics. Separate train/validation dataset roots prevent accidental cross-split chunks. Raw object state, contacts, stage, and outcome remain available for audit but are excluded from model observations.

Collection produced 10,256 observation/action rows over 1,025.6 simulated seconds: 8,188 training rows and 2,068 validation rows. Each camera recorded 5,140 frames. All 20 episodes passed physical grasp/contact, payload-weight, release-in-bin, and timing checks. Extra hover thrust was 0.980993–0.981013 N, consistent with the 100 g payload. Maximum camera mount position error was 1.11e-7 m.

The first collector attempt exposed a steady ~3.5 cm lateral loaded-flight bias. The M2 expert's 2.5 cm drone-centering rule timed out despite the block fitting over the bin. M3 now gates release on the measured block's conservative bounding sphere fitting inside the bin with 5 cm clearance, low drone speed, sufficient payload height, and both finger contacts. M2 retains its original gate. The initial successful/failed attempts and API probe remain in ignored `debug_initial_expert/`; final collection uses one consistent revised contract.

## LeRobot export and normalization

Uses the existing `lerobot` environment and its installed v0.6.2 dataset API: `create → add_frame → save_episode → finalize`. The [upstream dataset documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) describes this lifecycle. All data stays local, with Hugging Face offline mode enabled.

RGB is stored as lossless embedded images in Parquet for exact pixel checks. Rows are 10 Hz, so alternate camera images repeat intentionally. Each row also stores the original simulation, image capture, and availability timestamps under `recording.*`; these audit fields are filtered out by the model loader. Image tensors returned by LeRobot are float32 CHW in `[0,1]`.

`normalization.json` is computed only from training episodes. State statistics count each training row once; action statistics count each valid future target in each anchored training chunk, matching uniform sampling of chunk starts. Padding does not contribute. Constant dimensions use scale 1. Validation uses these same training statistics. Native LeRobot `meta/stats.json` describes **absolute disk actions** and must not replace the custom relative-chunk statistics.

M4 must wire this contract into π0 preprocessing exactly once and disable any second generic relative-action or normalization pass. `ChunkDataset` is currently the validated data adapter, not yet a complete π0 training integration. Image processing and model padding are M4 work.

## Reproduction commands

Run from `/home/user/pivla`:

```bash
# Collect all scenarios; existing passed episodes are preserved on resume.
bash scripts/run_isaac.sh scripts/m3_collect.py --enable_cameras

# Export to a NEW destination. Existing dataset roots are never overwritten.
bash scripts/run_lerobot.sh -m airvla_isaac.data.convert_to_lerobot

# Validate raw alignment, split isolation, statistics, images, and chunk batches.
bash scripts/run_lerobot.sh -m airvla_isaac.data.validate_dataset

# Encode the five representative videos and plot action reconstruction.
bash scripts/run_lerobot.sh scripts/m3_previews.py

# Simulator-independent contract checks.
bash scripts/run_lerobot.sh scripts/test_action_codec.py
bash scripts/run_lerobot.sh scripts/test_dataset_statistics.py
```

The simulator uses `isaaclab`; export and loading use `lerobot`. Both launchers mount shared installations read-only, set imports only for the child process, and redirect caches into the workspace. No package installation, model download, or upload is required. The package initializer now leaves simulation registration explicit so dataset utilities can import without Isaac Sim; M0 already imports its tasks explicitly.

The existing LeRobot environment has no Matplotlib. The preview script therefore reuses `isaaclab` Python's installed plotting stack inside the same read-only sandbox. This is CPU plotting/video encoding and does not start Isaac Sim or modify either environment.

## Evidence and scope

Raw episodes live in `artifacts/M3/episode_NNN/`; each has `samples.jsonl`, 200 Hz `flight.jsonl`, camera calibration/timestamps, PNG streams, scenario configuration, and outcome. The full plan is in local `collection_manifest.json`. Exported datasets live in `datasets/airvla_m3/train` and `datasets/airvla_m3/validation`.

Commit only the compact schema, split manifest, normalization, validation/export summaries, environment comparison, report, preview manifest, and reconstruction plot. Raw episodes, dataset files, caches, videos, and frame contact sheets are ignored by Git.

This small dataset tests the pipeline and modest scene variations. A single held-out layout is a development check, not a broad generalization benchmark. It does not establish learned policy performance; no policy has been trained.
