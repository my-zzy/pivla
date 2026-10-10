# AirVLA simulation project

M1 hover and camera checks passed on 2026-10-10. Start with [implement_guide.md](implement_guide.md); the next milestone is **M2: physical grasp-and-lift**.

- [M0 results, environment details, and commands](artifacts/M0/report.md)
- [M1 results, video, camera preview, and plots](artifacts/M1/report.md)
- [Recorded versions and source commits](configs/versions.yaml)
- [Technical reference](AirVLA_IsaacLab_Reproduction_Guide.md)
- [Generated external project](airvla_isaac/README.md)

The existing `isaaclab` and `lerobot` environments are shared with other projects. Do not upgrade them, install this project into them, change their editable links, or edit their source checkouts. The launcher supplies project-local imports without an installation:

```bash
cd /home/user/pivla
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras
```

To watch live in an Isaac Sim desktop window, add `--gui --keep-open`:

```bash
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras --gui --keep-open
```

The viewport starts inside the room looking at the drone. After the test, the final scene stays open for inspection until you close the window. The saved [three-camera video](artifacts/M1/hover.mp4) can also be opened without starting Sim.

The launcher uses the existing `bwrap` executable to mount host files read-only, except this workspace and the GPU's device/kernel interfaces. M1 caches and logs go under `artifacts/M1/runtime/`; M0 commands retain `artifacts/M0/runtime/`. `/tmp` and `/var/tmp` are temporary private mounts. No shell profile or global environment variable is changed. In a restricted agent sandbox, this command needs host GPU access. Ordinary Ubuntu terminals may also need the one-time [AppArmor permission for bwrap](docs/launcher-troubleshooting.md); the launcher checks this before starting Sim.

The drone implementation is under `airvla_isaac/source/airvla_isaac/airvla_isaac/sim/`, with settings in [configs/sim.yaml](configs/sim.yaml). The original cartpole task remains as an M0 setup check. No dataset, policy training, or model download has run. Use the existing environment named `lerobot` for dataset/policy work, as requested. It currently lacks the `transformers` dependency required by its π0 extra; resolve training readiness before M4. No environment clone is planned, and selecting this environment does not change its packages.
