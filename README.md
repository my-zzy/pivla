# AirVLA simulation project

M0–M4 are complete, and M5 has measured the M4 debug checkpoint in closed-loop simulation. Expert replay succeeded, but π0 achieved **0/2 familiar and 0/4 held-out successes** despite low offline prediction error. The action conversion and timing audit passed. Next is improving training/data coverage before RTC; start with [implement_guide.md](implement_guide.md).

- [M0 results, environment details, and commands](artifacts/M0/report.md)
- [M1 results, video, camera preview, and plots](artifacts/M1/report.md)
- [M2 physical pick-and-place, load measurements, and commands](artifacts/M2/report.md)
- [M3 dataset contract, collection, export, and validation](artifacts/M3/report.md)
- [M4 training results, prediction plots, and checkpoint](artifacts/M4/report.md)
- [M5 closed-loop failures, successful expert replay, videos, and timing](artifacts/M5/report.md)
- [Recorded versions and source commits](configs/versions.yaml)
- [Technical reference](AirVLA_IsaacLab_Reproduction_Guide.md)
- [Generated external project](airvla_isaac/README.md)

The existing `isaaclab` and `lerobot` environments are shared with other projects. The user authorized adding the dependencies needed for M4 to `lerobot`; ten missing packages were installed while preserving all 111 existing package versions. Keep `isaaclab`, editable links, and shared source checkouts unchanged. The launcher supplies project-local imports without installing this project:

```bash
cd /home/user/pivla
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras
```

For one headless physical pick-and-place demo and its saved video:

```bash
bash scripts/run_isaac.sh scripts/m2_pick_place.py --enable_cameras --trials 1 --output artifacts/M2/demo
MPLCONFIGDIR=/home/user/pivla/artifacts/M2/plot_cache \
  /home/user/miniconda3/envs/isaaclab/bin/python -B scripts/m2_summarize.py --output artifacts/M2/demo
```

Open `artifacts/M2/demo/pick_place.mp4` afterward; video playback does not run the simulator. The ten-trial validation uses `--trials 10` with the default output `artifacts/M2/`, and records cameras for the first trial. Add `--record-all` to record every trial.

M3 uses the existing environments through separate launchers:

```bash
# Simulation: twenty varied episodes, with cameras for every episode.
bash scripts/run_isaac.sh scripts/m3_collect.py --enable_cameras
# Dataset work: local/offline LeRobot export, checks, and five preview videos.
bash scripts/run_lerobot.sh -m airvla_isaac.data.convert_to_lerobot
bash scripts/run_lerobot.sh -m airvla_isaac.data.validate_dataset
bash scripts/run_lerobot.sh scripts/m3_previews.py
```

The dataset contract is in [configs/dataset.yaml](configs/dataset.yaml). Export creates `datasets/airvla_m3/{train,validation}` and refuses to overwrite existing datasets. `airvla_isaac.data.chunk_dataset.ChunkDataset` supplies 50-step relative actions with episode-end masks and training-only normalization. Keep absolute actions on disk, and use this adapter's statistics rather than the native absolute-action statistics for future training.

To watch live in an Isaac Sim desktop window, add `--gui --keep-open`:

```bash
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras --gui --keep-open
```

The viewport starts inside the room looking at the drone. After the test, the final scene stays open for inspection until you close the window. The saved [three-camera video](artifacts/M1/hover.mp4) can also be opened without starting Sim.

The launcher uses the existing `bwrap` executable to mount host files read-only, except this workspace and the GPU's device/kernel interfaces. Caches and logs go under each milestone's `artifacts/M<number>/runtime/`. `/tmp` and `/var/tmp` are temporary private mounts. No shell profile or global environment variable is changed. In a restricted agent sandbox, this command needs host GPU access. Ordinary Ubuntu terminals may also need the one-time [AppArmor permission for bwrap](docs/launcher-troubleshooting.md); the launcher checks this before starting Sim.

The drone implementation is under `airvla_isaac/source/airvla_isaac/airvla_isaac/sim/`, with baseline settings in [configs/sim.yaml](configs/sim.yaml) and M2 overrides in [configs/manipulation.yaml](configs/manipulation.yaml). The scripted expert, recorders, exporter, and loader live in `data/`; the action codec and π0 adapter are in `policy/`. The original cartpole task remains as an M0 setup check. Dataset and policy work uses the existing `lerobot` environment. M4 trained the π0 action expert on two demonstrations for 1,600 steps, reaching 8.04 cm position-command RMSE and 98.12% gripper accuracy with 10.708 GiB peak CUDA allocation. The checkpoint at `artifacts/M4/debug_refine/checkpoint/` reloads with identical predictions; see the [M4 report](artifacts/M4/report.md). M5 evaluated this checkpoint in closed loop with 0/6 successes; full baseline training remains pending. No environment clone was needed.
