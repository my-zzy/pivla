# AirVLA simulation project

M0–M2 are complete: setup, hover/cameras, and physical pick-and-place passed on 2026-10-10. M2 succeeded in 10/10 fixed-scene trials. Start with [implement_guide.md](implement_guide.md); next is **M3: dataset and action conversion**.

- [M0 results, environment details, and commands](artifacts/M0/report.md)
- [M1 results, video, camera preview, and plots](artifacts/M1/report.md)
- [M2 physical pick-and-place, load measurements, and commands](artifacts/M2/report.md)
- [Recorded versions and source commits](configs/versions.yaml)
- [Technical reference](AirVLA_IsaacLab_Reproduction_Guide.md)
- [Generated external project](airvla_isaac/README.md)

The existing `isaaclab` and `lerobot` environments are shared with other projects. Do not upgrade them, install this project into them, change their editable links, or edit their source checkouts. The launcher supplies project-local imports without an installation:

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

To watch live in an Isaac Sim desktop window, add `--gui --keep-open`:

```bash
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras --gui --keep-open
```

The viewport starts inside the room looking at the drone. After the test, the final scene stays open for inspection until you close the window. The saved [three-camera video](artifacts/M1/hover.mp4) can also be opened without starting Sim.

The launcher uses the existing `bwrap` executable to mount host files read-only, except this workspace and the GPU's device/kernel interfaces. Caches and logs go under each milestone's `artifacts/M<number>/runtime/`. `/tmp` and `/var/tmp` are temporary private mounts. No shell profile or global environment variable is changed. In a restricted agent sandbox, this command needs host GPU access. Ordinary Ubuntu terminals may also need the one-time [AppArmor permission for bwrap](docs/launcher-troubleshooting.md); the launcher checks this before starting Sim.

The drone implementation is under `airvla_isaac/source/airvla_isaac/airvla_isaac/sim/`, with baseline settings in [configs/sim.yaml](configs/sim.yaml) and M2 overrides in [configs/manipulation.yaml](configs/manipulation.yaml). The scripted expert and raw recorder live in the package's `data/` directory. The original cartpole task remains as an M0 setup check. M2 raw episodes are not yet a LeRobot dataset; policy training and model downloads have not started. Use the existing environment named `lerobot` for dataset/policy work, as requested. It currently lacks the `transformers` dependency required by its π0 extra; resolve training readiness before M4. No environment clone is planned, and selecting this environment does not change its packages.
