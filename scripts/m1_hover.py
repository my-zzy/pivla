#!/usr/bin/env python3
"""Run M1 physics, reset, hover and RGB checks; save reproducible evidence."""
import argparse
import json
import shlex
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=ROOT / 'configs/sim.yaml')
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/M1')
parser.add_argument('--smoke', action='store_true', help='Two-second debugging run; does not certify M1.')
parser.add_argument('--keep-open', action='store_true', help='Keep the GUI open on the final scene after the test.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error('M1 needs --enable_cameras')
if args.keep_open and args.headless:
    parser.error('--keep-open requires the launcher --gui option')
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from airvla_isaac.sim.env import HoverEnv
from airvla_isaac.sim.flight_controller import rotation_matrix


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    config = yaml.safe_load(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'frames').mkdir(exist_ok=True)
    save_json(output / 'result.json', {'status': 'running'})
    save_json(output / 'config.json', config)
    env = HoverEnv(config, args.device)
    checks = {}

    # Verify the actual PhysX mass/inertia, not just values authored to USD.
    actual_mass = env.drone.root_physx_view.get_masses().cpu().numpy()
    actual_inertia = env.drone.root_physx_view.get_inertias().cpu().numpy().reshape(-1, 3, 3)
    checks['mass_inertia'] = bool(
        np.allclose(actual_mass, config['drone']['mass_kg']) and
        np.allclose(actual_inertia[0], np.diag(config['drone']['inertia_kg_m2']), atol=1e-6))

    # A 0.15-second unpowered fall must accelerate under gravity.
    start = env.reset(config['seed'])[0].cpu().numpy()
    for _ in range(round(.15 / env.dt)):
        falling = env.step(apply_control=False)[0].cpu().numpy()
    fall = {'duration_s': env.time, 'altitude_drop_m': float(start[2] - falling[2]),
            'final_vz_m_s': float(falling[9]), 'expected_vz_m_s': -config['gravity'] * env.time}
    checks['unpowered_gravity'] = bool(fall['altitude_drop_m'] > .08 and
        abs(fall['final_vz_m_s'] - fall['expected_vz_m_s']) < .03)

    # Perturb the physical state by stepping between resets, then compare three
    # same-seed initial states AND their one-second controlled trajectories.
    initial_states, trajectories = [], []
    for trial in range(3):
        initial_states.append(env.reset(config['seed'])[0].cpu().tolist())
        trajectory = []
        for _ in range(config['physics_hz']):
            trajectory.append(env.step()[0].cpu().tolist())
        trajectories.append(trajectory)
    initial_delta = float(np.max(np.abs(np.array(initial_states) - initial_states[0])))
    trajectory_delta = float(np.max(np.abs(np.array(trajectories) - trajectories[0])))
    checks['same_seed_initial_state'] = initial_delta <= 1e-7
    checks['same_seed_trajectory'] = trajectory_delta <= 1e-5
    different = env.reset(config['seed'] + 1)[0].cpu().numpy()
    checks['different_seed_jitter'] = bool(np.linalg.norm(different[:3] - np.array(initial_states[0][:3])) > 1e-4)
    save_json(output / 'reset_checks.json', {
        'seed': config['seed'], 'initial_states': initial_states,
        'max_initial_difference': initial_delta, 'max_trajectory_difference': trajectory_delta,
        'unpowered_fall': fall, 'mass_kg': actual_mass.tolist(), 'inertia_kg_m2': actual_inertia.tolist(),
        'checks': checks,
    })
    print('M1 physics/reset checks:', checks, flush=True)
    if not all(checks.values()):
        raise RuntimeError('Physics/reset checks failed; stopping before hover. See reset_checks.json.')

    initial = env.reset(config['seed'])[0].cpu().numpy()
    env.warmup_rendering()
    duration = 2. if args.smoke else config['settle_s'] + config['hover_s']
    rows, camera_records, statistics = [], [], {name: [] for name in env.cameras}
    mount_errors, axis_errors = [], []
    capture_start = time.perf_counter()
    with (output / 'flight.jsonl').open('w') as flight_log, (output / 'cameras.jsonl').open('w') as camera_log:
        for step in range(round(duration / env.dt)):
            state = env.step()[0].cpu().numpy()
            row = {'time_s': env.time, 'position_w': state[:3].tolist(), 'quaternion_wxyz': state[3:7].tolist(),
                   'velocity_w': state[7:10].tolist(), 'angular_velocity_w': state[10:13].tolist(),
                   'target_position_w': config['target']['position'], 'target_yaw': config['target']['yaw'],
                   'force_b_n': env.force[0, 0].cpu().tolist(), 'torque_b_nm': env.torque[0, 0].cpu().tolist(),
                   'control_time_s': ((env.step_count-1) // env.control_interval) * env.control_interval * env.dt}
            flight_log.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if not np.isfinite(state).all() or state[2] < .1 or np.linalg.norm(state[:2]) > 2.8:
                raise RuntimeError(f'Drone left the safe test volume at {env.time}s: {state}')
            if env.step_count % env.camera_interval:
                continue
            frames = env.capture()
            frame_id = len(camera_records)
            mosaic = Image.new('RGB', (3 * config['cameras']['width'], config['cameras']['height'] + 40), '#172132')
            draw = ImageDraw.Draw(mosaic)
            record = {'frame': frame_id, 'capture_time_s': env.time, 'available_time_s': env.time,
                      'available_wall_elapsed_s': time.perf_counter() - capture_start, 'cameras': {}}
            body_r = rotation_matrix(torch.tensor(state[None, 3:7]))[0].numpy()
            for index, (name, frame) in enumerate(frames.items()):
                rgb = frame.pop('rgb')
                if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
                    raise RuntimeError(f'Unexpected {name} RGB shape/dtype: {rgb.shape}, {rgb.dtype}')
                statistics[name].append(float(rgb.std()))
                image = Image.fromarray(rgb)
                directory = output / 'frames' / name
                directory.mkdir(exist_ok=True)
                image.save(directory / f'{frame_id:04d}.png')
                image.save(output / f'{name}.png')
                mosaic.paste(image, (index * rgb.shape[1], 40))
                draw.text((index * rgb.shape[1] + 10, 8), f'{name.upper()}  |  t={env.time:.1f}s', fill='white')
                if name != 'external':
                    mount = config['cameras'][name]
                    expected_position = state[:3] + body_r @ np.array(mount['position'])
                    expected_r = body_r @ rotation_matrix(torch.tensor([mount['quaternion']]))[0].numpy()
                else:
                    expected_position = np.array(config['cameras']['external']['position'])
                    expected_r = None
                mount_errors.append(float(np.linalg.norm(np.array(frame['position_w']) - expected_position)))
                if expected_r is not None:
                    actual_r = rotation_matrix(torch.tensor([frame['quaternion_world_wxyz']]))[0].numpy()
                    axis_errors.append(float(np.max(np.abs(actual_r - expected_r))))
                record['cameras'][name] = frame
            mosaic.save(output / 'frames' / f'{frame_id:04d}.png')
            mosaic.save(output / 'camera_preview.png')
            camera_records.append(record)
            camera_log.write(json.dumps(record, allow_nan=False) + '\n')
            if env.step_count % config['physics_hz'] == 0:
                error = np.linalg.norm(state[:3] - config['target']['position'])
                print(f'M1 t={env.time:5.1f}s | position error={error:.5f}m | thrust={row["force_b_n"][2]:.3f}N', flush=True)

    times = np.array([row['time_s'] for row in rows])
    positions = np.array([row['position_w'] for row in rows])
    quats = np.array([row['quaternion_wxyz'] for row in rows])
    yaw = np.arctan2(2*(quats[:, 0]*quats[:, 3] + quats[:, 1]*quats[:, 2]),
                     1 - 2*(quats[:, 2]**2 + quats[:, 3]**2))
    yaw_error = np.arctan2(np.sin(yaw - config['target']['yaw']), np.cos(yaw - config['target']['yaw']))
    errors = np.linalg.norm(positions - config['target']['position'], axis=1)
    hover_mask = times > config['settle_s']
    camera_times = np.array([r['capture_time_s'] for r in camera_records])
    checks.update({
        'exact_physics_step_count': env.physics_steps == len(rows),
        'finite_state_and_wrench': bool(np.isfinite(np.array([r['force_b_n'] + r['torque_b_nm'] for r in rows])).all()
                                       and np.isfinite(positions).all() and np.isfinite(quats).all()),
        'camera_rgb_nonuniform': all(min(values) > 5. for values in statistics.values()),
        'camera_mount_positions': max(mount_errors) < 1e-4,
        'camera_mount_rotations': max(axis_errors) < 1e-4,
        'camera_timing': bool(np.allclose(np.diff(camera_times), 1/config['camera_hz']) and
                              len(camera_records) == round(duration * config['camera_hz'])),
    })
    metrics = {'initial_position_error_m': float(np.linalg.norm(initial[:3] - config['target']['position'])),
               'max_camera_mount_position_error_m': max(mount_errors), 'max_camera_rotation_matrix_error': max(axis_errors),
               'minimum_rgb_std': {k: min(v) for k, v in statistics.items()}}
    if not args.smoke:
        metrics.update({
            'hover_max_position_error_m': float(errors[hover_mask].max()),
            'hover_rms_position_error_m': float(np.sqrt(np.mean(errors[hover_mask]**2))),
            'hover_max_yaw_error_rad': float(np.abs(yaw_error[hover_mask]).max()),
            'hover_mean_thrust_n': float(np.array([r['force_b_n'][2] for r in rows])[hover_mask].mean()),
            'hover_sample_count': int(hover_mask.sum()),
        })
        checks['hover_20_seconds'] = int(hover_mask.sum()) == round(config['hover_s'] * config['physics_hz'])
        checks['hover_position'] = metrics['hover_max_position_error_m'] <= config['position_tolerance_m']
        checks['hover_yaw'] = metrics['hover_max_yaw_error_rad'] <= config['yaw_tolerance_rad']
    command = ['bash', 'scripts/run_isaac.sh', 'scripts/m1_hover.py', '--enable_cameras',
               '--config', str(args.config), '--output', str(output)]
    if env.sim.has_gui():
        command.append('--gui')
    if args.smoke:
        command.append('--smoke')
    if args.keep_open:
        command.append('--keep-open')
    result = {'status': ('smoke_passed' if args.smoke else 'passed') if all(checks.values()) else 'failed',
              'simulation_duration_s': duration, 'physics_steps': len(rows), 'camera_frames_per_stream': len(camera_records),
              'physics_engine_steps': env.physics_steps, 'gui': bool(env.sim.has_gui()),
              'checks': checks, 'metrics': metrics,
              'timestamp_convention': 'Episode-relative simulation time. Rendering is synchronous with physics paused; capture and availability simulation times coincide.',
              'command': shlex.join(command)}
    save_json(output / 'result.json', result)
    print('M1 RESULT', json.dumps(result), flush=True)
    if not all(checks.values()):
        raise RuntimeError('M1 validation failed; see result.json')
    if args.keep_open and env.sim.has_gui():
        print('Test finished. Final scene held for inspection; close the window to exit.', flush=True)
        while app.is_running():
            env.render()
            time.sleep(.02)


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except BaseException as exc:
        # Sim's immediate shutdown can swallow an exception and return exit 0.
        # Flush evidence and preserve failure status in this bounded subprocess.
        import os
        import sys
        import traceback
        traceback.print_exc()
        # Never leave a failed attempt labelled "running" or retain an older pass.
        result_path = args.output.resolve() / 'result.json'
        try:
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save_json(result_path, result)
        except (OSError, ValueError):
            pass  # Preserve the original exception if evidence cannot be written.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close(skip_cleanup=True)
