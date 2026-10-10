# AirVLA Isaac Lab external project

Generated from the existing Isaac Lab v2.3.2 direct-workflow template. The current task, `Template-Airvla-Isaac-Direct-v0`, is the one-environment cartpole smoke test used to complete M0. The drone implementation starts in M1.

Run from the parent workspace without installing into the shared environment:

```bash
cd /home/user/pivla
timeout --kill-after=15 240 bash scripts/run_isaac.sh scripts/m0_project_smoke.py
```

The launcher supplies this package and the selected Isaac Lab checkout through a process-local `PYTHONPATH`. Shared environments and source directories are mounted read-only; runtime caches stay in the workspace. Do not run `pip install -e` or the template dependency installer in the existing environments.

Source is under `source/airvla_isaac/airvla_isaac/`. The M0 runner selects the local cartpole USD under `assets/m0/`; the task uses a procedural floor. These are setup fixtures to replace as the drone task is implemented.

See the [working checklist](../implement_guide.md), [M0 report](../artifacts/M0/report.md), and [recorded versions](../configs/versions.yaml).
