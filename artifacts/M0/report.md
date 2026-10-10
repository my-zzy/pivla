# M0 setup report

**Status: passed — 2026-10-09 (Asia/Shanghai).** The next milestone is M1. No installation is needed to begin simulator work.

## Verified checks

| Check | Result | Evidence |
|:--|:--|:--|
| Empty-scene tutorial | 120 GPU physics steps; process exited successfully | [empty.log](empty.log) |
| Camera tutorial | 20 RGB frames at 640 × 480; dimensions/nonuniform pixels checked; preview visually inspected | [preview](camera_preview.png), [validation](camera_validation.json), [camera.log](camera.log) |
| External direct-workflow project | One cartpole environment, 120 steps, finite observations/rewards; process exited successfully | [result](project_result.json), [project.log](project.log) |
| Isaac Python CUDA | Small tensor operation passed on RTX 5090 | [CUDA result](cuda_isaaclab.json) |
| LeRobot Python CUDA | Small tensor operation passed on RTX 5090 | [CUDA result](cuda_lerobot.json) |
| Shared environment preservation | Package metadata fingerprints and source checkout status match before/after | [before](inventory_before.json), [after](inventory_after.json) |

The camera test uses the upstream tutorial's resolution. Three task cameras at 256 × 256 belong to M1. The direct-workflow example establishes the project integration; it does not establish drone dynamics or flight control.

## Machine and existing software

| Item | Observed |
|:--|:--|
| OS | Ubuntu 24.04.4 LTS, x86_64 |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB reported memory |
| Driver | 580.173.02 |
| CPU / RAM | Ryzen 9 9950X3D; about 60.5 GiB usable RAM reported by Kit |
| Free workspace disk | About 1,493 GiB at inventory |
| Simulator interpreter | `/home/user/miniconda3/envs/isaaclab/bin/python`, Python 3.11.15 |
| Simulator | Isaac Sim package 5.1.0.0; selected Isaac Lab checkout v2.3.2 |
| Simulator PyTorch | 2.7.0+cu128; CUDA operation passed |
| Training interpreter | `/home/user/miniconda3/envs/lerobot/bin/python`, Python 3.12.14 |
| LeRobot | 0.6.2, editable source at `/home/user/lerobot` |
| Training PyTorch | Runtime 2.11.0+cu130; CUDA operation passed |

Full package lists and exact commits are in the inventories and [versions.yaml](../../configs/versions.yaml). GPU runtime versions need not be identical because the two interpreters remain separate.

## How the shared installations were protected

- No pip/conda install, upgrade, uninstall, environment activation-file edit, or shell-profile edit was performed.
- The standard `isaaclab.sh --new` wrapper installs dependencies. Instead, `scripts/prepare_m0.py` called the already-installed generator directly. Automatic Git initialization/commit was disabled; only the new external project was generated.
- The shared Isaac environment's editable links point to `/home/user/workspace/objsplat_ws/.cache/IsaacLab` (extension metadata 0.54.4). Those links were preserved. This project's launcher selects `/home/user/IsaacLab` v2.3.2 (source extension 0.54.2) through a child-process-only `PYTHONPATH`.
- Simulator checks ran with shared filesystem paths mounted read-only. Only this workspace, private temporary directories, and GPU device/kernel interfaces were writable.
- Local package metadata fingerprints covered 3,497 files in `isaaclab` and 1,276 in `lerobot`; both matched after execution. All three inspected source checkouts retained their original commits and clean Git status.

## Local adaptations needed for reliable smoke checks

The original tutorials remain unchanged in `/home/user/IsaacLab`. Copies under `tutorials/` have finite loops and use Isaac Sim's supported immediate-shutdown option. The camera copy flushes its image writer before closing.

Initial attempts exposed wrapper/runtime issues, preserved in the `*_initial_*.log` files. CUDA works with access to the GPU driver's `/proc` interface; making that interface read-only caused error 304. Private `/var/tmp` prevents OptiX from trying to write shared cache files. Per-script runtime folders avoid cache locks between runs.

The native USD resolver could not download the tutorial's remote ground plane. The local camera copy and generated task use procedural floors. The small NVIDIA cartpole asset and its mesh dependency were downloaded over HTTPS into `airvla_isaac/assets/m0/`; their URLs, hashes, and dependencies are in [asset_manifest.json](asset_manifest.json). No global asset settings were changed.

## Reproduce the successful checks

Run these from `/home/user/pivla` in a normal terminal, one at a time. Environment activation and package installation are unnecessary. The launcher adds headless mode and isolates runtime output.

```bash
timeout --kill-after=15 180 bash scripts/run_isaac.sh artifacts/M0/tutorials/create_empty.py
timeout --kill-after=15 240 bash scripts/run_isaac.sh artifacts/M0/tutorials/run_usd_camera.py --enable_cameras --save
timeout --kill-after=15 240 bash scripts/run_isaac.sh scripts/m0_project_smoke.py
```

Successful runs print `M0 empty: completed 120 steps`, `M0 camera: completed 20 steps`, and `M0 project: ... 'status': 'passed'`. Check these markers and generated evidence as well as the process exit status; Kit can exit with code zero after some initialization errors.

## What to do next

Proceed to M1: implement the drone rigid body, thrust/torque controller, repeatable reset, and front/down/external cameras inside the external package. Keep one environment and primitive geometry initially.

The shared LeRobot environment has working CUDA but no `transformers`, which its π0 extra requires. This does not block M1–M3. Following the user's environment choice, use the existing environment named `lerobot` for dataset/policy work and resolve its missing π0 dependency before M4. No clone is planned and no shared packages have been changed. π0 checkpoint loading and training memory use remain untested.
