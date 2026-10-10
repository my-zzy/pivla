# AirVLA Isaac Lab external project

Generated from the existing Isaac Lab v2.3.2 direct-workflow template. The drone implementation is in `source/airvla_isaac/airvla_isaac/sim/`. The M1 runner verifies physical hover and three cameras; M2 adds physical sliding fingers and scripted pick-and-place. The expert and raw recorder are in `data/`. `Template-Airvla-Isaac-Direct-v0` remains the cartpole smoke test used to complete M0.

Run from the parent workspace without installing into the shared environment:

```bash
cd /home/user/pivla
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras
bash scripts/run_isaac.sh scripts/m2_pick_place.py --enable_cameras --trials 1 --output artifacts/M2/demo
```

The launcher supplies this package and the selected Isaac Lab checkout through a process-local `PYTHONPATH`. Shared environments and source directories are mounted read-only; runtime caches stay in the workspace. Do not run `pip install -e` or the template dependency installer in the existing environments.

The simulation modules separate scene geometry, drone mass/inertia, flight control, camera transforms, gripper contact/actuation, and stepping/reset logic. M2 extends `HoverEnv` through scene/asset factory methods and retains its one-tick stepping and capture implementation. Settings live in `../configs/sim.yaml` and `../configs/manipulation.yaml`. Both scenes use procedural geometry and need no asset downloads. Run `scripts/m0_project_smoke.py` through the same launcher when an installation regression check is needed; it uses the local cartpole assets under `assets/m0/`.

See the [working checklist](../implement_guide.md), [M2 report and commands](../artifacts/M2/report.md), [M1 report](../artifacts/M1/report.md), [M0 report](../artifacts/M0/report.md), and [recorded versions](../configs/versions.yaml).
