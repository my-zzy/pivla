#!/usr/bin/env python3
"""Run bounded, physical scripted pick/place episodes without changing environments."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=ROOT / 'configs/manipulation.yaml')
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/M2')
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--lift-only', action='store_true')
parser.add_argument('--record-all', action='store_true', help='Record camera streams for every trial instead of the first.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.trials < 1:
    parser.error('--trials must be positive')
app = AppLauncher(args).app

import numpy as np
import torch
import yaml
from airvla_isaac.sim.manipulation_env import ManipulationEnv
from airvla_isaac.data.expert_policy import ScriptedExpert
from airvla_isaac.data.collect_demos import EpisodeRecorder


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and key in result else value
    return result


def trial_metrics(rows, expert, config, env):
    task = config['task']
    stats = {}
    for label, stage in [('empty', 'approach'), ('loaded', 'loaded_hold')]:
        selected = [r for r in rows if r['stage'] == stage]
        if not selected:
            raise RuntimeError(f'No {label} hover measurements')
        # Integer sampling avoids counting a floating-point boundary twice.
        window = selected[-round(task['hover_measure_s'] * config['physics_hz']):]
        stats[label] = {
            'mean_thrust_n': float(np.mean([r['force_b_n'][2] for r in window])),
            'mean_altitude_m': float(np.mean([r['drone_state'][2] for r in window])),
            'max_position_error_m': float(max(np.linalg.norm(np.array(r['drone_state'][:3])-r['target_position_w']) for r in window)),
            'samples': len(window),
            'both_fingers_contact_fraction': float(np.mean([all(r['contacts']) for r in window])),
        }
    expected = task['block_mass_kg'] * config['gravity']
    delta = stats['loaded']['mean_thrust_n'] - stats['empty']['mean_thrust_n']
    lifted = [r for r in rows if r['stage'] in ('loaded_hold', 'transport')]
    final = rows[-1]
    checks = {
        'all_stages_finished': expert.done,
        'exact_physics_steps': env.physics_steps == len(rows),
        'physical_load_increases_thrust': abs(delta-expected) < task['expected_load_thrust_tolerance_n'],
        'empty_hover_tracking': stats['empty']['max_position_error_m'] < .03,
        'loaded_hover_tracking': stats['loaded']['max_position_error_m'] < task['loaded_position_tolerance_m'],
        'sustained_loaded_contact': stats['loaded']['both_fingers_contact_fraction'] > .98,
        'held_clear_of_table': min(r['block_state'][2] for r in lifted) > task['table_top_m'] + .30,
        'hover_measurement_duration': all(s['samples'] == round(task['hover_measure_s'] * config['physics_hz']) for s in stats.values()),
        'finite_states_and_wrenches': bool(np.isfinite(np.array([
            r['drone_state']+r['block_state']+r['force_b_n']+r['torque_b_nm'] for r in rows])).all()),
    }
    if not args.lift_only:
        checks['released_inside_bin'] = expert.in_bin(final) and not any(final['contacts']) and final['command_open']
    return {'checks': checks, 'hover': stats, 'load_thrust_increase_n': delta, 'expected_payload_weight_n': expected,
            'duration_s': env.time, 'physics_steps': env.physics_steps,
            'final_block_state': final['block_state'], 'events': expert.events}


def main():
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = merge(yaml.safe_load((ROOT/'configs/sim.yaml').read_text()), yaml.safe_load(args.config.read_text()))
    if config['physics_hz'] % config['action_hz']:
        raise ValueError('action_hz must divide physics_hz')
    action_interval = config['physics_hz']//config['action_hz']
    save(output/'config.json', config)
    command = ['bash', 'scripts/run_isaac.sh', 'scripts/m2_pick_place.py', '--config', str(args.config),
               '--output', str(output), '--trials', str(args.trials)]
    for enabled, flag in [(args.enable_cameras, '--enable_cameras'), (not args.headless, '--gui'),
                          (args.lift_only, '--lift-only'), (args.record_all, '--record-all')]:
        if enabled:
            command.append(flag)
    result = {'status': 'running', 'trials': [], 'milestone_complete': False,
              'task_instruction': 'Pick up the block from the table and place it in the bin.',
              'command': shlex.join(command)}
    save(output/'result.json', result)
    env = ManipulationEnv(config, args.device, args.enable_cameras)
    actual_masses = {'drone': env.drone.root_physx_view.get_masses().cpu().tolist(),
                     'fingers': [f.root_physx_view.get_masses().cpu().tolist() for f in env.gripper.fingers],
                     'block': env.block.root_physx_view.get_masses().cpu().tolist()}
    result['actual_masses_kg'] = actual_masses
    mass_check = (np.allclose(actual_masses['drone'], config['drone']['mass_kg'])
                  and np.allclose(actual_masses['fingers'], config['gripper']['finger_mass_kg'])
                  and np.allclose(actual_masses['block'], config['task']['block_mass_kg']))
    if not mass_check:
        raise RuntimeError(f'Unexpected physical masses: {actual_masses}')
    result['physical_masses_verified'] = True
    initial_states = []
    for trial in range(args.trials):
        env.reset(config['seed'])
        initial = env.observation()
        initial_states.append(initial['drone_state']+initial['block_state'])
        expert = ScriptedExpert(config, args.lift_only)
        recorder = EpisodeRecorder(output/f'episode_{trial:03d}', config)
        record_cameras = args.enable_cameras and (trial == 0 or args.record_all)
        if record_cameras:
            env.warmup_rendering()
        rows = []
        print(f'M2 trial {trial+1}/{args.trials}', flush=True)
        try:
            while not expert.done:
                if env.step_count % action_interval == 0:
                    target, opened = expert.update(env.time, env.observation())
                    if expert.done:
                        break
                    env.target_position[:] = env.target_position.new_tensor(target)
                    env.gripper.command(opened)
                env.step()
                obs = env.observation()
                state = np.array(obs['drone_state'])
                if not np.isfinite(state).all() or state[2] < .45 or state[2] > 2.5 or np.linalg.norm(state[:2]) > 2.5:
                    raise RuntimeError(f'Drone left safe volume: {obs}')
                rows.append(recorder.record(env, obs, expert.stage))
                if record_cameras and env.step_count % env.camera_interval == 0:
                    recorder.capture(env, expert.stage)
                if env.step_count % (config['physics_hz']*2) == 0:
                    print(f'  {env.time:.1f}s {expert.stage} z={state[2]:.3f} block_z={obs["block_state"][2]:.3f} '
                          f'aperture={obs["aperture_m"]:.3f} contacts={obs["contacts"]}', flush=True)
            summary = trial_metrics(rows, expert, config, env)
            summary['camera_frames_per_stream'] = recorder.frame_count
            if record_cameras:
                summary['checks']['camera_frame_count'] = recorder.frame_count == env.step_count//env.camera_interval
                summary['max_camera_mount_position_error_m'] = recorder.max_mount_position_error_m
                summary['max_camera_mount_rotation_error'] = recorder.max_mount_rotation_error
            summary['initial_state'] = initial_states[-1]
            summary['status'] = 'passed' if all(summary['checks'].values()) else 'failed'
            save(recorder.output/'result.json', summary)
            result['trials'].append(summary)
            save(output/'result.json', result)
            if summary['status'] != 'passed':
                raise RuntimeError(f'Trial {trial} checks failed: {summary["checks"]}')
        except Exception as exc:
            save(recorder.output/'failure.json', {'stage': expert.stage, 'time_s': env.time,
                 'error': str(exc), 'events': expert.events, 'last_observation': env.observation()})
            raise
        finally:
            recorder.close()
    result['same_seed_initial_state_max_difference'] = float(np.max(np.abs(np.array(initial_states)-initial_states[0])))
    result['status'] = 'passed'
    result['milestone_complete'] = (args.trials >= 10 and not args.lift_only and args.enable_cameras
                                    and result['same_seed_initial_state_max_difference'] < 1e-6)
    result['timestamp_convention'] = 'Episode-relative simulation time; synchronous rendering pauses physics.'
    save(output/'result.json', result)
    print('M2 RESULT', result['status'], 'milestone_complete=', result['milestone_complete'], flush=True)


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except BaseException as exc:
        traceback.print_exc()
        path = args.output.resolve()/'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            result.update(status='failed', milestone_complete=False, error=f'{type(exc).__name__}: {exc}')
            save(path, result)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close(skip_cleanup=True)
