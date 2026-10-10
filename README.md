# AirVLA simulation project

M0 setup checks passed on 2026-10-09. Start with [implement_guide.md](implement_guide.md); the next milestone is **M1: force-driven drone hover and three cameras**.

- [M0 results, environment details, and commands](artifacts/M0/report.md)
- [Recorded versions and source commits](configs/versions.yaml)
- [Technical reference](AirVLA_IsaacLab_Reproduction_Guide.md)
- [Generated external project](airvla_isaac/README.md)

The existing `isaaclab` and `lerobot` environments are shared with other projects. Do not upgrade them, install this project into them, change their editable links, or edit their source checkouts. The launcher supplies project-local imports without an installation:

```bash
cd /home/user/pivla
timeout --kill-after=15 240 bash scripts/run_isaac.sh scripts/m0_project_smoke.py
```

The launcher uses the existing `bwrap` executable to mount host files read-only, except this workspace and the GPU's device/kernel interfaces. Caches and logs go under `artifacts/M0/runtime/`; `/tmp` and `/var/tmp` are temporary private mounts. No shell profile or global environment variable is changed. In a restricted agent sandbox, this command needs host GPU access; it works from the normal machine terminal.

The generated task is a cartpole setup check, not the drone. No dataset, policy training, or model download has run. Use the existing environment named `lerobot` for dataset/policy work, as requested. It currently lacks the `transformers` dependency required by its π0 extra; resolve training readiness before M4. No environment clone is planned, and selecting this environment does not change its packages.
