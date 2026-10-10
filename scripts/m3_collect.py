#!/usr/bin/env python3
"""Collect seeded M3 scenarios with observations explicitly preceding actions."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'artifacts/M3')
parser.add_argument('--episode-ids', nargs='+', type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error('M3 collection requires --enable_cameras')
app = AppLauncher(args).app

import numpy as np
import torch
import yaml
from airvla_isaac.sim.scenario_env import ScenarioEnv
from airvla_isaac.data.expert_policy import ScriptedExpert
from airvla_isaac.data.collect_demos import EpisodeRecorder
from airvla_isaac.policy.action_codec import state_from_raw


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def merge(base, updates):
    base = copy.deepcopy(base)
    for key, value in updates.items():
        base[key] = merge(base[key], value) if isinstance(value, dict) and key in base else value
    return base


def make_manifest(schema, base):
    c = schema['collection']
    rng = np.random.default_rng(c['seed'])
    episodes = []
    for scenario in range(c['scenario_count']):
        block = np.array(base['task']['block_position'], dtype=float)
        block[:2] += rng.uniform(-1, 1, 2)*c['block_xy_jitter_m']
        bin_pos = np.array(base['task']['bin_position'], dtype=float)
        bin_pos[:2] += rng.uniform(-1, 1, 2)*c['bin_xy_jitter_m']
        split = 'train' if scenario in schema['split']['train_scenarios'] else 'validation'
        for repeat in range(c['episodes_per_scenario']):
            index = len(episodes)
            cfg = copy.deepcopy(base)
            cfg['seed'] = c['seed']+index
            cfg['task']['block_position'] = block.tolist()
            cfg['task']['bin_position'] = bin_pos.tolist()
            cfg['target']['position'] = [*block[:2], base['task']['flight_height_m']]
            initial = np.array(cfg['target']['position'])
            initial[:2] += rng.uniform(-1, 1, 2)*c['drone_xy_jitter_m']
            initial[2] += rng.uniform(-1, 1)*c['drone_z_jitter_m']
            cfg['drone']['initial_position'] = initial.tolist()
            cfg['drone']['initial_yaw'] = float(rng.uniform(*c['drone_initial_yaw_range_rad']))
            episodes.append({'episode_id': index, 'scenario_id': scenario, 'repeat': repeat,
                             'split': split, 'seed': cfg['seed'], 'config': cfg})
    return {'schema': schema, 'episodes': episodes}


def main():
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    schema = yaml.safe_load((ROOT/'configs/dataset.yaml').read_text())
    base = merge(yaml.safe_load((ROOT/'configs/sim.yaml').read_text()),
                 yaml.safe_load((ROOT/'configs/manipulation.yaml').read_text()))
    base['task']['release_footprint_margin_m'] = schema['collection']['release_footprint_margin_m']
    manifest = make_manifest(schema, base)
    path = out/'collection_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Existing collection manifest differs; choose a fresh --output directory')
    save(path, manifest)
    selected = args.episode_ids if args.episode_ids is not None else range(len(manifest['episodes']))
    env = ScenarioEnv(base, args.device, True)
    for index in selected:
        item = manifest['episodes'][index]
        dest = out/f'episode_{index:03d}'
        result_path = dest/'result.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'passed':
            print(f'M3 episode {index}: already passed, preserving recording', flush=True)
            continue
        if dest.exists():
            raise RuntimeError(f'Incomplete episode exists: {dest}; retain it under a debug name before retrying')
        env.configure(item['config'])
        env.warmup_rendering()
        expert = ScriptedExpert(item['config'])
        recorder = EpisodeRecorder(dest, item['config'])
        save(dest/'config.json', item['config'])
        save(dest/'metadata.json', {k:v for k,v in item.items() if k!='config'})
        print(f'M3 episode {index}: scenario {item["scenario_id"]}, split {item["split"]}, seed {item["seed"]}', flush=True)
        rows, sample_count, latest = [], 0, None
        try:
            with (dest/'samples.jsonl').open('w') as samples:
                while True:
                    if env.step_count % env.camera_interval == 0:
                        latest = recorder.capture(env, expert.stage)
                    if env.step_count % (base['physics_hz']//schema['fps']) == 0:
                        observation = env.observation()
                        target, opened = expert.update(env.time, observation)
                        if expert.done:
                            break
                        assert latest is not None and latest['available_time_s'] <= env.time+1e-9
                        row = {'time_s': env.time, 'state': state_from_raw(observation['drone_state'], observation['aperture_m']).tolist(),
                               'action': [*target, 0., 0., float(env.target_yaw[0]), float(not opened)],
                               'stage': expert.stage, 'camera': latest, 'privileged': observation}
                        samples.write(json.dumps(row, allow_nan=False)+'\n')
                        sample_count += 1
                        env.target_position[:] = env.target_position.new_tensor(target)
                        env.gripper.command(opened)
                    env.step()
                    obs = env.observation()
                    state = np.array(obs['drone_state'])
                    if not np.isfinite(state).all() or state[2] < .45 or state[2] > 2.5 or np.linalg.norm(state[:2]) > 2.5:
                        raise RuntimeError(f'Unsafe drone state: {state}')
                    rows.append(recorder.record(env, obs, expert.stage))
            holds = {name: [r for r in rows if r['stage']==stage][-800:]
                     for name,stage in [('empty','approach'),('loaded','loaded_hold')]}
            means = {k: float(np.mean([r['force_b_n'][2] for r in v])) for k,v in holds.items()}
            checks = {'expert_finished': expert.done, 'released_in_bin': expert.in_bin(obs) and not any(obs['contacts']),
                      'physics_ticks': env.physics_steps==len(rows),
                      'loaded_contact': all(all(r['contacts']) for r in holds['loaded']),
                      'payload_weight': abs(means['loaded']-means['empty']-.981)<.20,
                      'samples_at_10hz': sample_count==round(env.time*schema['fps'])}
            result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
                      'duration_s': env.time, 'physics_steps': env.physics_steps, 'samples': sample_count,
                      'camera_frames_per_stream': recorder.frame_count, 'events': expert.events,
                      'mean_thrust_n': means, 'max_camera_mount_position_error_m': recorder.max_mount_position_error_m,
                      'max_camera_mount_rotation_error': recorder.max_mount_rotation_error,
                      'final_block_state': obs['block_state'], 'outcome': 'success' if all(checks.values()) else 'failed'}
            save(result_path, result)
            print('M3 episode', index, result['status'], 'samples', sample_count, flush=True)
            if not all(checks.values()):
                raise RuntimeError(f'Episode checks failed: {checks}')
        except BaseException as exc:
            save(dest/'failure.json', {'error': str(exc), 'stage': expert.stage, 'time_s': env.time, 'events': expert.events})
            raise
        finally:
            recorder.close()
    completed = [i for i in range(len(manifest['episodes'])) if (out/f'episode_{i:03d}/result.json').exists()
                 and json.loads((out/f'episode_{i:03d}/result.json').read_text())['status']=='passed']
    save(out/'collection_result.json', {'status': 'passed' if len(completed)==len(manifest['episodes']) else 'partial',
                                      'completed_episode_ids': completed, 'planned_episodes': len(manifest['episodes'])})


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close(skip_cleanup=True)
